"""
Memory-Augmented Legal Chatbot - Production Version
Architecture: Intent Analyzer → Memory Layers → RAG → Response Generation → Guardrails
Author: Hemant Thakkar
"""

import os
import re
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum

import numpy as np
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer, util

# CONFIGURATION

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    GROQ_MODEL = "llama-3.1-8b-instant"
    CHROMA_PERSIST_DIR = "./chroma_db"
    EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"
    LONG_TERM_COLLECTION = "legal_knowledge"
    MEDIUM_TERM_COLLECTION = "session_summaries"
    USER_PROFILE_COLLECTION = "user_profiles"
    
    # Memory thresholds
    SHORT_TERM_MAX_TURNS = 20
    MEDIUM_TERM_SUMMARY_TRIGGER = 5


# GROQ CLIENT

class GrokClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.GROQ_API_KEY
        self.client = Groq(api_key=self.api_key)

    def generate(self, prompt: str, temperature: float = 0.4, max_tokens: int = 1500) -> str:
        try:
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a legal AI assistant for Indian law."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f" Groq API Error: {e}")
            return f"Error generating response: {str(e)}"


# VECTOR DATABASE MANAGER (Hybrid Search)
class VectorDBManager:
    """Handles vector + lexical retrieval over ChromaDB."""

    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
        print(f" Connected to ChromaDB at {Config.CHROMA_PERSIST_DIR}")

        # Create/get collections
        self.collections = {
            Config.LONG_TERM_COLLECTION: self.client.get_or_create_collection(
                name=Config.LONG_TERM_COLLECTION, metadata={"hnsw:space": "cosine"}
            ),
            Config.MEDIUM_TERM_COLLECTION: self.client.get_or_create_collection(
                name=Config.MEDIUM_TERM_COLLECTION, metadata={"hnsw:space": "cosine"}
            ),
            Config.USER_PROFILE_COLLECTION: self.client.get_or_create_collection(
                name=Config.USER_PROFILE_COLLECTION, metadata={"hnsw:space": "cosine"}
            ),
        }

        # count docs safely
        try:
            data = self.collections[Config.LONG_TERM_COLLECTION].get(include=["documents"])
            doc_count = len(data.get("documents", []))
        except Exception:
            doc_count = 0
        print(f" Legal knowledge base loaded with {doc_count} chunks.")

    # ------------------------------------------------------------------
    def normalize_query(self, query: str):
        """Normalize & extract section numbers from a user query."""
        q = query.lower().strip()
        q = re.sub(r'\bipc\b', 'indian penal code', q)
        nums = re.findall(r"\b(\d+[A-Za-z]?)\b", q)
        nums = [n.upper() for n in nums if len(n) <= 5]
        return q, nums

    # ------------------------------------------------------------------
    def add_document(self, collection_name: str, text: str, metadata: Dict):
        """Add a single document with embedding and clean section metadata."""
        section = metadata.get("section")
        if not section:
            m = re.search(r'\bsection\s*(\d+[A-Za-z]?)\b', text, flags=re.I)
            if m:
                section = m.group(1).upper()
            else:
                section = "UNKNOWN"
            metadata["section"] = section

        emb = self.embedding_model.encode(text).tolist()
        cid = str(uuid.uuid4())
        self.collections[collection_name].add(
            ids=[cid],
            embeddings=[emb],
            documents=[text],
            metadatas=[metadata]
        )

        # ------------------------------------------------------------------
    def search(self, collection_name: str, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid numeric + semantic + lexical search (safe and deterministic)."""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                print(f" Missing collection {collection_name}")
                return []
    
            # Normalize query
            q_norm, sec_nums = self.normalize_query(query)
            query_emb = self.embedding_model.encode(q_norm).tolist()
    
            docs = []
    
            # --- STEP 1: Direct section match (metadata or text) ---
            if sec_nums:
                raw = collection.get(include=["documents", "metadatas"])
                all_docs = raw.get("documents", [])
                all_meta = raw.get("metadatas", [])
    
                # Exact metadata hit
                for i, meta in enumerate(all_meta):
                    sec = str(meta.get("section", "")).upper()
                    if sec in [s.upper() for s in sec_nums]:
                        docs.append({
                            "text": all_docs[i],
                            "metadata": meta,
                            "score": 1.0
                        })
    
                # Numeric text scan fallback
                if not docs:
                    for i, doc in enumerate(all_docs):
                        doc_lower = doc.lower()
                        for n in sec_nums:
                            if f"section {n.lower()}" in doc_lower or re.search(rf'\b{n}\b', doc_lower):
                                docs.append({
                                    "text": doc,
                                    "metadata": all_meta[i],
                                    "score": 0.9
                                })
                                break
                            
                if docs:
                    print(f" Section match found ({', '.join(sec_nums)}): {len(docs)} docs")
    
            # --- STEP 2: Semantic vector search (always runs) ---
            try:
                res = collection.query(
                    query_embeddings=[query_emb],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                if res and res.get("documents") and res["documents"][0]:
                    distances = np.array(res["distances"][0], dtype=float)
                    dmin, dmax = float(distances.min()), float(distances.max())
                    sims = (dmax - distances) / (dmax - dmin + 1e-9)
                    for i, doc in enumerate(res["documents"][0]):
                        docs.append({
                            "text": doc,
                            "metadata": res["metadatas"][0][i],
                            "score": float(np.clip(sims[i], 0.0, 1.0))
                        })
            except Exception as e:
                print(" Vector query failed:", e)
    
            # --- STEP 3: Lexical (keyword) fallback ---
            if not docs:
                try:
                    raw = collection.get(include=["documents", "metadatas"])
                    all_docs = raw.get("documents", [])
                    all_meta = raw.get("metadatas", [])
                    q_tokens = set(re.findall(r'\w+', q_norm))
                    for i, doc in enumerate(all_docs):
                        if not doc:
                            continue
                        doc_lower = doc.lower()
                        doc_tokens = set(re.findall(r'\w+', doc_lower))
                        inter = len(q_tokens & doc_tokens)
                        union = len(q_tokens | doc_tokens)
                        score = inter / union if union else 0
                        if score > 0.15:
                            docs.append({
                                "text": doc,
                                "metadata": all_meta[i],
                                "score": score
                            })
                except Exception as e:
                    print(" Lexical fallback failed:", e)
    
            # --- STEP 4: Deduplicate and rank ---
            unique = {}
            for d in docs:
                meta = d.get("metadata") or {}
                sec = meta.get("section", "")
                key = sec or d["text"][:100]
                if key not in unique or unique[key]["score"] < d["score"]:
                    unique[key] = d
    
            ranked = sorted(unique.values(), key=lambda x: x["score"], reverse=True)[:top_k]
            secs = [d.get("metadata", {}).get("section") for d in ranked]
            scs = [round(d["score"], 3) for d in ranked]
            print(f" search('{query}') → {len(ranked)} docs; scores={scs}; secs={secs}")
    
            return ranked or []
    
        except Exception as e:
            print(f" search() failed hard: {e}")
            return []

# MEMORY LAYERS
class ShortTermMemory:
    """Stores recent conversation turns"""
    def __init__(self, max_turns=20):
        self.buffer = []
        self.max_turns = max_turns
        self.turn_count = 0

    def add_turn(self, user: str, bot: str):
        self.turn_count += 1
        self.buffer.append({
            "turn": self.turn_count,
            "user": user, 
            "bot": bot,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.buffer) > self.max_turns:
            self.buffer.pop(0)

    def get_recent_context(self, last_n: int = 5) -> str:
        recent = self.buffer[-last_n:]
        return "\n".join([f"User: {t['user']}\nBot: {t['bot']}" for t in recent])
    
    def get_all_turns(self) -> List[Dict]:
        return self.buffer
    
    def clear(self):
        self.buffer.clear()


class MediumTermMemory:
    """Periodically summarizes conversation and stores in vector DB"""
    def __init__(self, grok: GrokClient, db: VectorDBManager):
        self.grok = grok
        self.db = db
        self.summaries = []
        
    def should_summarize(self, turn_count: int) -> bool:
        return turn_count > 0 and turn_count % Config.MEDIUM_TERM_SUMMARY_TRIGGER == 0
    
    def create_incremental_summary(self, recent_turns: List[Dict]) -> str:
        convo = "\n".join([f"User: {t['user']}\nBot: {t['bot']}" for t in recent_turns])
        prompt = f"""Summarize this legal chat segment in 2-3 concise lines, focusing on:
- Key legal topics discussed
- Important sections mentioned
- User's primary concerns

Conversation:
{convo}

Summary:"""
        summary = self.grok.generate(prompt, temperature=0.3, max_tokens=150)
        return summary.strip()
    
    def store_summary(self, user_id: str, summary: str, turn_range: str):
        self.db.add_document(
            Config.MEDIUM_TERM_COLLECTION,
            text=summary,
            metadata={
                "user": user_id,
                "type": "session_summary",
                "turn_range": turn_range,
                "timestamp": datetime.now().isoformat()
            }
        )
        self.summaries.append({
            "summary": summary,
            "turn_range": turn_range,
            "timestamp": datetime.now().isoformat()
        })
        print(f" Medium-term summary stored: {turn_range}")
    
    def get_relevant_summaries(self, query: str, top_k: int = 2) -> List[Dict]:
        return self.db.search(Config.MEDIUM_TERM_COLLECTION, query, top_k=top_k)
    
    def get_all_summaries(self) -> List[Dict]:
        return self.summaries


class LongTermMemory:
    """Persistent legal knowledge base"""
    def __init__(self, db: VectorDBManager):
        self.db = db
        self.last_retrieved = []

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        docs = self.db.search(Config.LONG_TERM_COLLECTION, query, top_k=top_k)
        self.last_retrieved = docs
        return docs
    
    def get_last_retrieved(self) -> List[Dict]:
        return self.last_retrieved or []


# INTENT, REASONING, VALIDATOR
class IntentType(Enum):
    STATUTE_LOOKUP = "statute_lookup"
    GENERAL_QUERY = "general_query"
    FOLLOW_UP = "follow_up"

@dataclass
class UserIntent:
    intent_type: IntentType

class IntentAnalyzer:
    def analyze(self, query: str, has_context: bool = False) -> UserIntent:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["section", "ipc", "crpc", "article"]):
            return UserIntent(IntentType.STATUTE_LOOKUP)
        
        if has_context and any(word in query_lower for word in ["what about", "and", "also", "more", "tell me"]):
            return UserIntent(IntentType.FOLLOW_UP)
        
        return UserIntent(IntentType.GENERAL_QUERY)
    

class LegalReasoningAgent:
    """
    Combines all memory layers (short, medium, long term)
    into a clean, contextually filtered prompt for Groq.
    """

    def __init__(self, grok: GrokClient):
        self.grok = grok

    def _summarize_docs(self, docs: List[Dict], max_chars: int = 600) -> str:
        """Format and trim long-term memory docs for the prompt."""
        if not docs:
            return "No specific legal sections found."

        # Deduplicate by section
        seen = set()
        formatted = []
        for d in sorted(docs, key=lambda x: x.get("score", 0), reverse=True):
            sec = (d.get("metadata") or {}).get("section", "Unknown")
            if sec in seen:
                continue
            seen.add(sec)
            text = re.sub(r"\s+", " ", d.get("text", ""))[:max_chars]
            formatted.append(f"• Section {sec} — {text.strip()}")
            if len(formatted) >= 4:
                break

        return "\n".join(formatted)

    def _summarize_medium(self, summaries: List[Dict], max_items: int = 2) -> str:
        """Format medium-term summaries."""
        if not summaries:
            return "No previous session summaries found."
        lines = []
        for s in summaries[:max_items]:
            txt = re.sub(r"\s+", " ", s.get("text", s.get("summary", "")))[:250]
            lines.append(f"• {txt}")
        return "\n".join(lines)

    def generate_response(
        self,
        query: str,
        long_term_docs: List[Dict],
        medium_term_summaries: List[Dict],
        short_term_context: str,
    ) -> str:
        """
        Use all memory layers to produce a concise, factual, legally grounded answer.
        """

        # Filter high-score long-term docs (>0.4 preferred)
        high_score_docs = [d for d in long_term_docs if d.get("score", 0) >= 0.4]
        if not high_score_docs and long_term_docs:
            # fallback to best 1–2 docs if no strong ones
            high_score_docs = long_term_docs[:2]

        # Build memory sections
        long_term_text = self._summarize_docs(high_score_docs)
        medium_term_text = self._summarize_medium(medium_term_summaries)
        short_term_trimmed = (
            "\n".join(short_term_context.splitlines()[-12:])
            if short_term_context
            else "No recent context available."
        )

        # Build the structured prompt
        prompt = f"""
        You are a **Legal AI Assistant** specialized in Indian law.
        Use the retrieved legal knowledge and previous memory context below
        to answer accurately and concisely.

        ### Short-term memory (recent chat)
        {short_term_trimmed}

        ### Medium-term memory (session summaries)
        {medium_term_text}

        ### Long-term memory (retrieved legal sections)
        {long_term_text}

        ### User query
        {query}

        ### Instructions
        1. Use only the information above; do not invent new sections.
        2. Cite IPC/CrPC sections **only** if they appear in the long-term memory list.
        3. Avoid repeating the same text or memory content.
        4. If no relevant section is found, respond: 
        "I could not find that section in the current database."
        5. Keep the answer short and factual (3–5 sentences).

        Now draft your answer below:
        """

        # Generate with Groq
        response = self.grok.generate(prompt, temperature=0.25, max_tokens=600)

        # Clean minor artifacts
        response = re.sub(r"\n{2,}", "\n", response).strip()
        return response

class GuardrailsValidator:
    def __init__(self, emb_model: SentenceTransformer):
        self.emb = emb_model
        self.last_similarity = 0.0

    def validate(self, response: str, docs: List[Dict]) -> Tuple[bool, float]:
        # If there are no docs, allow answer but similarity=0.0
        if not docs:
            return True, 0.0

        # Build small doc corpus (top 3)
        doc_text = " ".join([d["text"][:400] for d in docs[:3]])
        resp_emb = self.emb.encode(response, convert_to_tensor=True)
        doc_emb = self.emb.encode(doc_text, convert_to_tensor=True)
        similarity = util.cos_sim(resp_emb, doc_emb).item()
        self.last_similarity = similarity

        # if response cites sections, ensure those sections exist in docs 
        cited = re.findall(r'section\s+(\d+[A-Za-z]?)', response.lower())
        citation_ok = True
        if cited:
            cited_clean = set(re.sub(r'\W+','',c).upper() for c in cited)
            available = set()
            for d in docs:
                md_sec = str(d.get('metadata', {}).get('section','') or '').upper()
                if md_sec:
                    available.add(md_sec)
                # also check in text
                for sec in cited_clean:
                    if f"section {sec.lower()}" in (d.get('text','').lower()):
                        available.add(sec)
            # citation_ok only if all cited sections appear in available
            citation_ok = cited_clean.issubset(available)

        # final decision: both semantic similarity threshold and citation match
        is_valid = (similarity > 0.34) and citation_ok

        # logging
        print(f" Validation: similarity={similarity:.3f}, citation_ok={citation_ok}, is_valid={is_valid}")
        return is_valid, similarity

    def get_last_similarity(self) -> float:
        return self.last_similarity


# MAIN CHATBOT

class MemoryAugmentedLegalChatbot:
    def __init__(self):
        print(" Initializing Memory-Augmented Legal Chatbot...")
        self.emb = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.db = VectorDBManager(self.emb)
        self.grok = GrokClient()
        
        # Memory layers
        self.short_term = ShortTermMemory(max_turns=Config.SHORT_TERM_MAX_TURNS)
        self.medium_term = MediumTermMemory(self.grok, self.db)
        self.long_term = LongTermMemory(self.db)
        
        # Other components
        self.intent_analyzer = IntentAnalyzer()
        self.reasoning_agent = LegalReasoningAgent(self.grok)
        self.validator = GuardrailsValidator(self.emb)
        
        print(" Chatbot initialized successfully!\n")

    def chat(self, query: str, user_id: str = "default_user") -> Dict:
        """Main chat function using all memory layers"""
        
        # 1. Analyze intent
        has_context = len(self.short_term.get_all_turns()) > 0
        intent = self.intent_analyzer.analyze(query, has_context)
        
        # 2. Retrieve from LONG-TERM memory
        long_term_docs = self.long_term.retrieve(query, top_k=5)
        
        # 3. Retrieve from MEDIUM-TERM memory
        medium_term_summaries = self.medium_term.get_relevant_summaries(query, top_k=2)
        
        # 4. Get SHORT-TERM context
        short_term_context = self.short_term.get_recent_context(last_n=5)
        
        # 5. Generate response using all memory layers
        response = self.reasoning_agent.generate_response(
            query=query,
            long_term_docs=long_term_docs,
            medium_term_summaries=medium_term_summaries,
            short_term_context=short_term_context
        )
        
        # 6. Validate response
        is_valid, similarity = self.validator.validate(response, long_term_docs)
        
        # 7. Store in SHORT-TERM memory
        self.short_term.add_turn(query, response)
        
        # 8. Check if MEDIUM-TERM summary needed
        turn_count = self.short_term.turn_count
        if self.medium_term.should_summarize(turn_count):
            recent_turns = self.short_term.get_all_turns()[-Config.MEDIUM_TERM_SUMMARY_TRIGGER:]
            summary = self.medium_term.create_incremental_summary(recent_turns)
            turn_range = f"Turns {turn_count - len(recent_turns) + 1}-{turn_count}"
            self.medium_term.store_summary(user_id, summary, turn_range)
        
        return {
            "response": response,
            "intent": intent.intent_type.value,
            "validation_score": similarity,
            "long_term_docs": long_term_docs,
            "medium_term_summaries": medium_term_summaries,
            "short_term_turns": self.short_term.get_all_turns()
        }
    
    def get_memory_states(self) -> Dict:
        """Get current state of all memory layers for UI display"""
        return {
            "short_term": {
                "turns": self.short_term.get_all_turns(),
                "count": len(self.short_term.get_all_turns())
            },
            "medium_term": {
                "summaries": self.medium_term.get_all_summaries(),
                "count": len(self.medium_term.get_all_summaries())
            },
            "long_term": {
                "last_retrieved": self.long_term.get_last_retrieved(),
                "count": len(self.long_term.get_last_retrieved())
            },
            "validation": {
                "last_similarity": self.validator.get_last_similarity()
            }
        }
    
    def end_session(self, user_id: str = "default_user"):
        """End session and create final summary"""
        if len(self.short_term.get_all_turns()) > 0:
            summary = self.medium_term.create_incremental_summary(self.short_term.get_all_turns())
            self.medium_term.store_summary(user_id, summary, "Final Session Summary")
        self.short_term.clear()
        print(" Session ended and memory cleared.")

