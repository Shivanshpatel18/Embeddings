from embeddings import EMBEDDING_MODEL
import weaviate
import json


client = weaviate.Client(
    url = "https://weav-v9odgtgv.weaviate.network",  # Replace with your endpoint
    auth_client_secret=weaviate.auth.AuthApiKey(api_key="ntNylMr3IwCQ0OUIPAJKIzdwdK7fPVjPQxwT"),  # Replace w/ your Weaviate instance API key
)
print(client.is_ready())

client.schema.delete_all()

class_obj = {"class": "DocumentSearch", "vectorizer": "none"}
client.schema.create_class(class_obj)

# Test source documents
documents = [
    "A group of vibrant parrots chatter loudly, sharing stories of their tropical adventures.",
    "The mathematician found solace in numbers, deciphering the hidden patterns of the universe.",
    "The robot, with its intricate circuitry and precise movements, assembles the devices swiftly.",
    "The chef, with a sprinkle of spices and a dash of love, creates culinary masterpieces.",
    "The ancient tree, with its gnarled branches and deep roots, whispers secrets of the past.",
    "The detective, with keen observation and logical reasoning, unravels the intricate web of clues.",
    "The sunset paints the sky with shades of orange, pink, and purple, reflecting on the calm sea.",
    "In the dense forest, the howl of a lone wolf echoes, blending with the symphony of the night.",
    "The dancer, with graceful moves and expressive gestures, tells a story without uttering a word.",
    "In the quantum realm, particles flicker in and out of existence, dancing to the tunes of probability."]

client.batch.configure(batch_size=len(documents))

def generate_embeddings(txt):
    emb = EMBEDDING_MODEL.get_text_embedding(
        txt
    )
    return emb

with client.batch as batch:
    for i, doc in enumerate(documents):
        properties = {"source_text": doc}
        vector = generate_embeddings(doc)
        batch.add_data_object(properties, "DocumentSearch", vector=vector)


query = "Give me some content about the birds"
query_vector = generate_embeddings(query)
# print(query_vector)
# print(len(query_vector))
result = client.query.get("DocumentSearch", ["source_text"]).with_near_vector({
    "vector": query_vector,
    "certainty": 0.7
}).with_limit(1).with_additional(['certainty', 'distance']).do()

print(json.dumps(result, indent=4))