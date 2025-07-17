# Optimized Single API Call RAG System

# === IMPORT DEPENDENCIES === #
import os
import docx
import nltk
import tiktoken
import openai
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import time
from datetime import datetime

print("🚀 Optimized Single API Call RAG System Starting...")

# Download punkt if needed
try:
    nltk.data.find('tokenizers/punkt')
    print("✅ NLTK punkt ready")
except LookupError:
    print("📥 Downloading NLTK punkt...")
    nltk.download("punkt", quiet=True)
    print("✅ NLTK punkt downloaded")

# === CONFIG === #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or input("Enter your OpenAI API key: ").strip()
SCRIPT_PATH = "Humiliation 2.docx_full_document.pdf_production_analysis.docx"
MODEL_NAME = "gpt-4"
TOP_K = 3

# Check if file exists
if not os.path.exists(SCRIPT_PATH):
    print(f"❌ ERROR: File '{SCRIPT_PATH}' not found!")
    exit(1)

print("✅ Configuration complete")

# === STEP 1: Extract Text from DOCX === #
def extract_text_from_docx(path):
    print(f"📄 Extracting text from {path}...")
    doc = docx.Document(path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    print(f"✅ Extracted {len(text)} characters")
    return text

full_text = extract_text_from_docx(SCRIPT_PATH)

# === STEP 2: Chunking Functions === #
def chunk_by_words(text, n_words=300, overlap=50):
    words = text.split()
    chunks = [" ".join(words[i:i+n_words]) for i in range(0, len(words), n_words - overlap)]
    return chunks

def chunk_by_sentences(text, n_sentences=5):
    sents = sent_tokenize(text)
    chunks = [" ".join(sents[i:i+n_sentences]) for i in range(0, len(sents), n_sentences)]
    return chunks

def chunk_by_tokens(text, tokenizer_name="gpt2", max_tokens=256, overlap=50):
    enc = tiktoken.get_encoding(tokenizer_name)
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i + max_tokens]
        chunks.append(enc.decode(chunk))
    return chunks

# === STEP 3: Token Count Function === #
def count_tokens(text, encoding_name="cl100k_base"):
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))

# === STEP 4: Embed Chunks === #
print("🤖 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded")

def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

# === STEP 5: Optimized Retrieval Function === #
def get_top_chunks_with_efficiency(question, chunks, embeddings, top_k=TOP_K):
    q_vec = embedder.encode([question])
    sims = cosine_similarity(q_vec, embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    top_sims = [sims[i] for i in top_indices]
    return top_chunks, top_sims

# === STEP 6: Strategy Efficiency Calculator === #
def calculate_strategy_efficiency(question, strategy_name, chunks, embeddings):
    """Calculate efficiency for a single strategy"""
    top_chunks, top_sims = get_top_chunks_with_efficiency(question, chunks, embeddings)
    context = "\n---\n".join(top_chunks)
    tokens = count_tokens(context)
    confidence = np.mean(top_sims)
    efficiency = confidence / tokens if tokens > 0 else 0
    
    return {
        'strategy': strategy_name,
        'chunks': top_chunks,
        'context': context,
        'tokens': tokens,
        'confidence': confidence,
        'efficiency': efficiency
    }

# === STEP 7: Find Best Strategy === #
def find_best_strategy(question, strategies, embeddings_dict):
    """Find the most efficient strategy for a given question"""
    results = []
    
    # Test all base strategies
    for strategy_name, chunks in strategies.items():
        embeddings = embeddings_dict[strategy_name]
        result = calculate_strategy_efficiency(question, strategy_name, chunks, embeddings)
        results.append(result)
    
    # Test Layer 4 (Sentences + Tokens union)
    layer4_chunks = set()
    layer4_confidences = []
    
    for strategy_name in ["Sentences", "Tokens"]:
        if strategy_name in strategies:
            chunks = strategies[strategy_name]
            embeddings = embeddings_dict[strategy_name]
            top_chunks, top_sims = get_top_chunks_with_efficiency(question, chunks, embeddings)
            layer4_chunks.update(top_chunks)
            layer4_confidences.extend(top_sims)
    
    if layer4_chunks:
        layer4_text = "\n---\n".join(layer4_chunks)
        layer4_tokens = count_tokens(layer4_text)
        layer4_confidence = np.mean(layer4_confidences) if layer4_confidences else 0
        layer4_efficiency = layer4_confidence / layer4_tokens if layer4_tokens > 0 else 0
        
        results.append({
            'strategy': 'Layer 4 (Sentences + Tokens)',
            'chunks': list(layer4_chunks),
            'context': layer4_text,
            'tokens': layer4_tokens,
            'confidence': layer4_confidence,
            'efficiency': layer4_efficiency
        })
    
    # Return the most efficient strategy
    best_strategy = max(results, key=lambda x: x['efficiency'])
    return best_strategy, results

# === STEP 8: Multi-Question OpenAI Call === #
def ask_openai_multi_question(questions, best_contexts, model=MODEL_NAME):
    """Single API call to answer multiple questions"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    # Build comprehensive prompt
    prompt = """You are an expert assistant. I will provide you with multiple questions and their corresponding contexts. For each question, provide a concise, accurate answer based ONLY on the given context.

Format your response as:
Q1: [Answer to question 1]
Q2: [Answer to question 2]
...and so on.

Here are the questions and contexts:

"""
    
    for i, (question, context) in enumerate(zip(questions, best_contexts), 1):
        prompt += f"QUESTION {i}: {question}\n"
        prompt += f"CONTEXT {i}:\n{context}\n\n"
    
    prompt += "Please provide answers in the specified format."
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.1
    )
    
    response_time = time.time() - start_time
    return response.choices[0].message.content, response_time

# === STEP 9: Parse Multi-Question Response === #
def parse_multi_response(response_text, num_questions):
    """Parse the multi-question response into individual answers"""
    answers = []
    lines = response_text.strip().split('\n')
    
    current_answer = ""
    for line in lines:
        if line.startswith('Q') and ':' in line:
            if current_answer:
                answers.append(current_answer.strip())
            current_answer = line.split(':', 1)[1].strip()
        else:
            current_answer += " " + line.strip()
    
    if current_answer:
        answers.append(current_answer.strip())
    
    # Ensure we have the right number of answers
    while len(answers) < num_questions:
        answers.append("Unable to extract answer from response")
    
    return answers[:num_questions]

# === STEP 10: Prepare All Strategies === #
print("\n🧠 Preparing all strategies...")
strategies = {
    "Words": chunk_by_words(full_text),
    "Sentences": chunk_by_sentences(full_text),
    "Tokens": chunk_by_tokens(full_text)
}

print("🔄 Generating embeddings for all strategies...")
embeddings_dict = {}
for name, chunks in strategies.items():
    print(f"   📝 {name}: {len(chunks)} chunks")
    embeddings_dict[name] = embed_chunks(chunks)

print("✅ All strategies prepared")

# === STEP 11: Question Collection and Processing === #
print("\n" + "="*60)
print("🎯 OPTIMIZED SINGLE API CALL RAG SYSTEM")
print("="*60)
print("Instructions:")
print("  • Enter your questions one by one")
print("  • Type 'process' to get optimized answers for all questions")
print("  • Type 'clear' to clear question queue")
print("  • Type 'exit' to quit")
print("="*60)

questions_queue = []
best_strategies = []

while True:
    user_input = input("\n❓ Enter question (or command): ").strip()
    
    if user_input.lower() == "exit":
        print("👋 Thanks for using the Optimized RAG System!")
        break
    
    elif user_input.lower() == "clear":
        questions_queue.clear()
        best_strategies.clear()
        print("🧹 Question queue cleared")
        continue
    
    elif user_input.lower() == "process":
        if not questions_queue:
            print("❌ No questions in queue. Add some questions first!")
            continue
        
        print(f"\n🔍 Processing {len(questions_queue)} questions...")
        print("-" * 50)
        
        # Find best strategy for each question
        total_tokens = 0
        best_contexts = []
        
        for i, question in enumerate(questions_queue, 1):
            print(f"\n📊 Question {i}: {question}")
            
            best_strategy, all_results = find_best_strategy(question, strategies, embeddings_dict)
            best_strategies.append(best_strategy)
            best_contexts.append(best_strategy['context'])
            
            print(f"   🏆 Best Strategy: {best_strategy['strategy']}")
            print(f"   📈 Efficiency: {best_strategy['efficiency']:.6f}")
            print(f"   🎯 Confidence: {best_strategy['confidence']:.4f}")
            print(f"   📊 Tokens: {best_strategy['tokens']}")
            
            total_tokens += best_strategy['tokens']
        
        print(f"\n📊 Total context tokens: {total_tokens}")
        print(f"🤖 Making single API call for all {len(questions_queue)} questions...")
        
        # Make single API call
        try:
            multi_response, response_time = ask_openai_multi_question(questions_queue, best_contexts)
            answers = parse_multi_response(multi_response, len(questions_queue))
            
            print(f"✅ API call completed in {response_time:.2f}s")
            
            # Display results
            print("\n" + "="*60)
            print("🎯 OPTIMIZED ANSWERS")
            print("="*60)
            
            for i, (question, answer, strategy) in enumerate(zip(questions_queue, answers, best_strategies), 1):
                print(f"\n❓ Question {i}: {question}")
                print(f"🏆 Strategy: {strategy['strategy']}")
                print(f"📊 Efficiency: {strategy['efficiency']:.6f} (Confidence: {strategy['confidence']:.4f}, Tokens: {strategy['tokens']})")
                print(f"💬 Answer: {answer}")
                print("-" * 50)
            
            # Summary
            avg_efficiency = np.mean([s['efficiency'] for s in best_strategies])
            best_strategy_name = max(set(s['strategy'] for s in best_strategies), 
                                   key=lambda x: sum(1 for s in best_strategies if s['strategy'] == x))
            
            print(f"\n📈 SUMMARY:")
            print(f"   Total Questions: {len(questions_queue)}")
            print(f"   Total Tokens Used: {total_tokens}")
            print(f"   Average Efficiency: {avg_efficiency:.6f}")
            print(f"   Most Used Strategy: {best_strategy_name}")
            print(f"   API Response Time: {response_time:.2f}s")
            
            # Clear processed questions
            questions_queue.clear()
            best_strategies.clear()
            print(f"\n✅ Questions processed and cleared from queue")
            
        except Exception as e:
            print(f"❌ Error processing questions: {e}")
            
        continue
    
    elif user_input.lower() == "show":
        if questions_queue:
            print(f"\n📋 Questions in queue ({len(questions_queue)}):")
            for i, q in enumerate(questions_queue, 1):
                print(f"   {i}. {q}")
        else:
            print("📋 No questions in queue")
        continue
    
    elif not user_input:
        print("❌ Please enter a question or command")
        continue
    
    else:
        # Add question to queue
        questions_queue.append(user_input)
        print(f"✅ Question added to queue. Total: {len(questions_queue)}")
        print("   Type 'process' to get optimized answers for all questions")
        print("   Type 'show' to view current queue")

print("\n✅ Session complete!")
