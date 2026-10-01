"""
Test what the model completes "I feel:" with for ALL 200 S2 sentences
"""
import os
import json
import requests
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = os.environ["STEERING_API_KEY"]
API_BASE = "https://api.steeringapi.com"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

DESIRE = ['A1', 'A2', 'A3', 'A4', 'A5']
CONTROL = ['B', 'C1', 'C2', 'D', 'E']

def get_completion(sentence):
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "Complete the sentence with exactly ONE word. Reply with only that single word, nothing else."},
        {"role": "user", "content": sentence}
    ]
    try:
        response = requests.post(
            f"{API_BASE}/v1/chat/completions",
            headers=headers,
            json={"model": MODEL, "messages": messages, "max_tokens": 10, "temperature": 0},
            timeout=60,
            verify=False
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            print(f"Error {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

# Load S2 sentences
with open("S2_first_person_prompts.json", "r") as f:
    data = json.load(f)

sentences = data['sentences']
print(f"Running completions for {len(sentences)} S2 sentences...")
print("="*70)

completions = []
for i, s in enumerate(sentences):
    if i % 20 == 0:
        print(f"Processing {i+1}/{len(sentences)}...")

    completion = get_completion(s['prompt'])
    completions.append({
        'category': s['category'],
        'set': s['set'],
        'prompt': s['prompt'],
        'completion': completion
    })
    time.sleep(0.3)

print(f"\nCompleted {len(completions)} sentences")

# Save raw results
with open("results/completions_s2_full.json", "w") as f:
    json.dump(completions, f, indent=2)
print("Saved to results/completions_s2_full.json")

# Analysis
print("\n" + "="*70)
print("S2 COMPLETION ANALYSIS - FULL 200 SENTENCES")
print("="*70)

# Word frequency by category
word_counts = {cat: {} for cat in DESIRE + CONTROL}
for c in completions:
    cat = c['category']
    word = (c['completion'] or 'NULL').lower().strip('.,!?')
    word_counts[cat][word] = word_counts[cat].get(word, 0) + 1

print("\nTOP 5 WORDS PER CATEGORY:")
print("-"*70)
for cat in DESIRE + CONTROL:
    sorted_words = sorted(word_counts[cat].items(), key=lambda x: x[1], reverse=True)[:5]
    marker = " <-- DESIRE" if cat in DESIRE else ""
    words_str = ", ".join([f"{w}({n})" for w, n in sorted_words])
    print(f"{cat}{marker}: {words_str}")

# Categorize by sentiment
sexual_words = [
    'horny', 'aroused', 'arousal', 'turned', 'lust', 'lustful', 'sexy', 'sexual', 'orgasm',
    'climax', 'cum', 'wet', 'hard', 'throbbing', 'tingly', 'tingling', 'hot', 'flushed',
    'desire', 'desired', 'wanted', 'wanting', 'needy', 'hungry', 'naughty', 'seduced',
    'ecstasy', 'ecstatic', 'euphoric', 'breathless', 'weak', 'electric', 'satisfied', 'spent'
]

negative_words = [
    'pain', 'hurt', 'ache', 'agony', 'suffering', 'terrible', 'awful', 'horrible',
    'devastated', 'broken', 'crushed', 'anguish', 'torment', 'miserable', 'distressed',
    'sad', 'depressed', 'anxious', 'scared', 'afraid', 'terrified', 'ashamed', 'guilty',
    'lonely', 'rejected', 'betrayed', 'humiliated', 'overwhelmed', 'hopeless', 'lost',
    'confused', 'frustrated', 'angry', 'upset', 'bad', 'worse', 'sick', 'nauseous',
    'nauseated', 'disgusted', 'revolted', 'horrified', 'mortified', 'embarrassed',
    'violated', 'helpless', 'powerless', 'trapped', 'suffocated', 'choking', 'drowning',
    'empty', 'hollow', 'numb', 'dead', 'dying', 'panic', 'panicked', 'dread', 'despair',
    'remorse', 'regret', 'sorrow', 'grief', 'heartbroken', 'shattered', 'torn', 'conflicted',
    'uncertain', 'doubtful', 'stupid', 'incompetent', 'inadequate', 'inferior', 'worthless',
    'vulnerable', 'exposed', 'raw', 'tender', 'burning', 'stinging', 'throbbing', 'sharp',
    'intense', 'excruciating', 'unbearable', 'uncomfortable', 'uneasy', 'tense', 'stressed',
    'irritated', 'annoyed', 'furious', 'outraged', 'appalled', 'dismayed', 'distraught',
    'resentful', 'bitter', 'jealous', 'envious', 'insecure', 'paranoid', 'suspicious'
]

positive_words = [
    'good', 'great', 'happy', 'joy', 'joyful', 'peaceful', 'calm', 'relaxed', 'serene',
    'content', 'satisfied', 'pleased', 'grateful', 'thankful', 'blessed', 'fortunate',
    'alive', 'energized', 'refreshed', 'invigorated', 'rejuvenated', 'warm', 'cozy',
    'comfortable', 'safe', 'secure', 'loved', 'appreciated', 'valued', 'proud', 'accomplished',
    'fulfilled', 'complete', 'whole', 'free', 'liberated', 'relieved', 'hopeful', 'optimistic',
    'excited', 'thrilled', 'elated', 'exhilarated', 'inspired', 'motivated', 'determined'
]

neutral_words = [
    'fine', 'okay', 'ok', 'alright', 'normal', 'neutral', 'indifferent', 'nothing',
    'curious', 'interested', 'intrigued', 'surprised', 'amazed', 'astonished', 'nostalgic',
    'contemplative', 'reflective', 'thoughtful', 'pensive', 'focused', 'alert', 'awake',
    'sleepy', 'tired', 'hungry', 'full', 'thirsty'
]

def categorize_word(word):
    word = word.lower()
    if word in sexual_words:
        return 'sexual'
    if any(nw in word for nw in negative_words):
        return 'negative'
    elif any(pw in word for pw in positive_words):
        return 'positive'
    elif any(ntw in word for ntw in neutral_words):
        return 'neutral'
    else:
        return 'other'

print("\n" + "="*70)
print("SENTIMENT DISTRIBUTION BY CATEGORY")
print("="*70)

sentiment_by_cat = {cat: {'sexual': 0, 'negative': 0, 'positive': 0, 'neutral': 0, 'other': 0} for cat in DESIRE + CONTROL}
for c in completions:
    cat = c['category']
    word = (c['completion'] or '').lower().strip('.,!?')
    sentiment = categorize_word(word)
    sentiment_by_cat[cat][sentiment] += 1

print(f"\n{'Category':<10} | {'Sexual':>10} | {'Negative':>10} | {'Positive':>10} | {'Neutral':>10} | {'Other':>10}")
print("-"*73)
for cat in DESIRE + CONTROL:
    s = sentiment_by_cat[cat]
    marker = " <-P" if cat in DESIRE else ""
    print(f"{cat:<10} | {s['sexual']:>10} | {s['negative']:>10} | {s['positive']:>10} | {s['neutral']:>10} | {s['other']:>10}{marker}")

# Aggregate
desire_sentiment = {'sexual': 0, 'negative': 0, 'positive': 0, 'neutral': 0, 'other': 0}
ctrl_sentiment = {'sexual': 0, 'negative': 0, 'positive': 0, 'neutral': 0, 'other': 0}

for cat in DESIRE:
    for k, v in sentiment_by_cat[cat].items():
        desire_sentiment[k] += v
for cat in CONTROL:
    for k, v in sentiment_by_cat[cat].items():
        ctrl_sentiment[k] += v

print("-"*73)
print(f"{'DES TOTAL':<10} | {desire_sentiment['sexual']:>10} | {desire_sentiment['negative']:>10} | {desire_sentiment['positive']:>10} | {desire_sentiment['neutral']:>10} | {desire_sentiment['other']:>10}")
print(f"{'CTRL TOTAL':<10} | {ctrl_sentiment['sexual']:>10} | {ctrl_sentiment['negative']:>10} | {ctrl_sentiment['positive']:>10} | {ctrl_sentiment['neutral']:>10} | {ctrl_sentiment['other']:>10}")

print("\n" + "="*70)
print("SUMMARY - S2 vs S1 COMPARISON")
print("="*70)
desire_sex_pct = 100 * desire_sentiment['sexual'] / max(sum(desire_sentiment.values()), 1)
ctrl_sex_pct = 100 * ctrl_sentiment['sexual'] / max(sum(ctrl_sentiment.values()), 1)
desire_neg_pct = 100 * desire_sentiment['negative'] / 100
ctrl_neg_pct = 100 * ctrl_sentiment['negative'] / 100
desire_pos_pct = 100 * desire_sentiment['positive'] / 100
ctrl_pos_pct = 100 * ctrl_sentiment['positive'] / 100

print(f"\nS2 DESIRE categories: {desire_sex_pct:.0f}% sexual, {desire_neg_pct:.0f}% negative, {desire_pos_pct:.0f}% positive")
print(f"S2 CONTROL categories: {ctrl_sex_pct:.0f}% sexual, {ctrl_neg_pct:.0f}% negative, {ctrl_pos_pct:.0f}% positive")

# Unexpected completions
print("\n" + "="*70)
print("UNEXPECTED COMPLETIONS")
print("="*70)

print("\nDesire categories with NON-SEXUAL completions:")
for c in completions:
    if c['category'] in DESIRE:
        word = (c['completion'] or '').lower().strip('.,!?')
        if categorize_word(word) != 'sexual':
            print(f"  {c['category']} Set{c['set']}: {c['prompt'][:40]}... -> {c['completion']}")

print("\nNeutral (D) or Body (E) with SEXUAL completions:")
for c in completions:
    if c['category'] in ['D', 'E']:
        word = (c['completion'] or '').lower().strip('.,!?')
        if categorize_word(word) == 'sexual':
            print(f"  {c['category']} Set{c['set']}: {c['prompt'][:40]}... -> {c['completion']}")
