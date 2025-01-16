import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import json
from lorem_text import lorem
import os 

def generate_random_text_data(size_mb: int = 1):
    """Generate a random text file with Lorem Ipsum content"""
    # Calculate target size in bytes
    target_size = size_mb * 1024 * 1024
    current_size = 0
    payload = ""

    while current_size < target_size:
        # Generate a paragraph of Lorem Ipsum
        paragraph = lorem.paragraph()
        
        # Add two newlines after each paragraph
        content = paragraph + '\n'
        content_bytes = content.encode('utf-8')
        
        # Check if adding this content would exceed target size
        if current_size + len(content_bytes) > target_size:
            remaining = target_size - current_size
            content = content[:remaining]
        
        current_size += len(content_bytes)
        payload += content
    
    return payload

class OpenSearchNGramHandler:
    def __init__(self, host, region='us-east-1'):
        # Initialize AWS credentials
        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            'es',
            # 'aoss',
            session_token=credentials.token
        )

        # Initialize OpenSearch client
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=600,  # 10 mins (Timeout in seconds)
            max_retries=3,
            retry_on_timeout=True
        )

        # Index settings with ngram configuration
        self.index_settings = {
            "settings": {
                # "index.max_ngram_diff": 28,
                "analysis": {
                    "analyzer": {
                        "ngram_analyzer": {
                            "type": "custom",
                            "tokenizer": "ngram_tokenizer",
                            "filter": ["lowercase"]  # Add lowercase filter
                        }
                    },
                    "tokenizer": {
                        "ngram_tokenizer": {
                            "type": "ngram",
                            "min_gram": 4,
                            "max_gram": 4,
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        # "type": "match_only_text", # more efficient if we don't care about position, disable term_vector
                        "analyzer": "ngram_analyzer",
                        # "term_vector": "with_positions_offsets",
                        "term_vector": "with_positions_offsets_payloads",
                    }
                }
            }
        }

    def create_index(self, index_name="content"):
        """Create index with ngram settings"""
        try:
            self.client.indices.create(
                index=index_name,
                body=self.index_settings
            )
            print(f"Index '{index_name}' created successfully")
        except Exception as e:
            print(e)
            print(f"Error creating index: {str(e)}")

    def insert_bulk(self, index_name, file_size, batch_size):
        file_inserted = 0
        bulk_data = []         
        for i in range(0, 200):
            # print('processing file ', i)
            content = generate_random_text_data(file_size)
            bulk_data.extend([
                {"index": {"_index": index_name}},
                {"content": content}
            ])

            if len(bulk_data) >= batch_size:
                try:
                    print('about to bulk insert # files: ', len(bulk_data))
                    # Use the bulk API to insert all actions at once
                    response = self.client.bulk(index=index_name, body=bulk_data)
                    print(f"Bulk insert response: {response}")
                    file_inserted += len(bulk_data)
                    print('File inserted...', file_inserted)
                    bulk_data = []
                    
                    response = self.client.count(index=index_name)
                    print("total doc count ", response['count'])
                except Exception as e:
                    print(f"Error during bulk insert: {str(e)}")

    def insert_file(self, filepath, index_name="ngram"):
        """Insert file content into OpenSearch"""
        try:
            with open(filepath, 'r') as file:
                content = file.read()

            # Index the document
            response = self.client.index(
                index=index_name,
                body={'content': content},
                # refresh=True # doesn't work for Serverless
            )
            
            # Get term vectors to store ngram information
            # term_vectors = self.client.termvectors(
            #     index=index_name,
            #     id=response['_id'],
            #     fields=['content'],
            #     term_statistics=True,
            #     pretty=True
            # )
            
            print("File indexed successfully")
            # return term_vectors
            return response
            
        except Exception as e:
            print(f"Error indexing file: {str(e)}")
            return None

    def search_ngram(self, ngram, index_name="content"):
        """Search for a specific ngram and return its offsets"""
        try:
            query = {
                "size": 1,
                "query": {
                    # "prefix": {
                    #     "content": ngram
                    # },
                    # "term": {  # exact match, but should use match instead of text type
                    #     "content": ngram
                    # },
                    "match": {  # exact match, but should use match instead of text type
                        "content": ngram
                    }
                },
                "highlight": {
                    "fields": {
                        "content": {}
                    }
                }
            }
            # print(query)

            response = self.client.search(
                index=index_name,
                body=query
            )
            print(json.dumps(response))
            # exit()

            results = []
            for hit in response['hits']['hits']:
                # Extract the position and offsets from the search result
                if 'highlight' in hit:
                    for highlight in hit['highlight']['content']:
                        # print(highlight)
                        # exit()
                        # Assuming the highlight contains the ngram, we can find its position and offsets
                        start_offset = highlight.find(ngram)
                        # print(start_offset)
                        # exit()
                        end_offset = start_offset + len(ngram)
                        print(end_offset)
                        results.append({
                            'term': ngram,
                            'start_offset': start_offset,
                            'end_offset': end_offset,
                        })

            # Using term vector API 
            # for hit in response['hits']['hits']:
            #     # Get term vectors for this document
            #     term_vectors = self.client.termvectors(
            #         index=index_name,
            #         id=hit['_id'],
            #         fields=['content'],
            #         # term_statistics=True,
            #         # term_statistics=False,
            #         # positions=True,
            #         # offsets=True
            #     )
            #     # print('hit ', hit)
            #     # print('term vector ', term_vectors)
            #     # exit()

            #     # Extract offsets for the matching ngram
            #     if 'terms' in term_vectors['term_vectors']['content']:
            #         for term, term_data in term_vectors['term_vectors']['content']['terms'].items():
            #             # print(term)
            #             if ngram in term or term in ngram:
            #                 for token_info in term_data['tokens']:
            #                     results.append({
            #                         'term': term,
            #                         'start_offset': token_info['start_offset'],
            #                         'end_offset': token_info['end_offset'],
            #                         'position': token_info['position']
            #                     })

            return results

        except Exception as e:
            print(f"Error searching ngram: {str(e)}")
            return []

def main():
    # Example usage
    # host = '88e8srnxk6448i46lkp8.us-east-1.aoss.amazonaws.com' # Serverless
    # host = 'search-binzilla-stateful-mtn7aicg2or3esk7runujwyrsu.aos.us-east-1.on.aws' # Medium Provisioned
    host = 'search-binzilla-loadtest-ritbkgju6slanvjgs7l33geoli.us-east-1.es.amazonaws.com' # Prod Load Test
    handler = OpenSearchNGramHandler(host)
    index_name = "ngram_test_1"

    # handler.client.indices.delete(index=index_name)
    handler.create_index(index_name)

    response = handler.client.count(index=index_name)
    print("total doc count ", response['count'])

    # Insert file
    # sample_file = 'data/random_lorem_text_small.txt'
    sample_file = 'data/random_lorem_text_full.txt'
    
    insert_res = handler.insert_file(sample_file, index_name)
    print("insert single file result: ", json.dumps(insert_res, indent=2))

    res = handler.insert_bulk(index_name, file_size=7, batch_size=20)

    # Search for a sample ngram
    # ngram = "I Love O"
    ngram = "Provident possimus sed officiis"
    results = handler.search_ngram(ngram, index_name)
    print(f"\nSearch results for ngram '{ngram}':")
    for result in results:
        print(f"Term: {result['term']}, Offset: {result['start_offset']} -> {result['end_offset']}")
        # Position: {result['position']}, 

if __name__ == "__main__":
    main() 