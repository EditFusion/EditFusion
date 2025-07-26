import requests
from requests.exceptions import ConnectionError, SSLError, Timeout
from urllib3.exceptions import ProtocolError
from datetime import datetime, timedelta
import time
import random
import string

id = f"repos_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}"
output_csv = f"output/{id}.csv"
with open(output_csv, "w") as f:
    f.write("primaryLanguage,name,url,assignableUsers,isFork,isInOrganization,stargazerCount,createdAt,pushedAt,diskUsage\n")

 # GitHub API URL and your GitHub personal access tokens
url = "https://api.github.com/graphql"
tokens = [
]
# Randomly select a token
def get_random_token():
    return random.choice(tokens)

lowThreshold = 100  # Minimum number of stars for repositories
highThreshold = 1000000  # Maximum number of stars for repositories

 # GraphQL query
def get_query_with_stars_less_than_and_date_within(curr, query_start_date, query_end_date):
    # YYYY-MM-DDTHH:MM:SS+00:00
    fmt_start_date = query_start_date.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    fmt_end_date = query_end_date.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return """
    query ($afterCursor: String) {
    search(
        type: REPOSITORY,
        query: "stars:""" + str(lowThreshold) + ".." + str(curr) + " created:" + fmt_start_date + ".." + fmt_end_date + """ pushed:>=2022-10-28",
        first: 100,
        after: $afterCursor
    ) {
        pageInfo {
        endCursor
        hasNextPage
        }
        repos: edges {
        repo: node {
            ... on Repository {
            primaryLanguage {
                name
            }
            name
            url
            createdAt
            pushedAt
            assignableUsers {
                totalCount
            }
            isFork
            isInOrganization
            stargazerCount
            diskUsage
            }
        }
        }
    }
    }
    """

def convert_to_git_protocol_url(https_url):
    if https_url.startswith("https://github.com/"):
        git_url = https_url.replace("https://github.com/", "git@github.com:") + ".git"
        return git_url
    else:
        raise ValueError("Invalid GitHub HTTPS URL")

 # Initialize cursor and pagination control
after_cursor = None
has_next_page = True
total = 0

 # Retry mechanism configuration
MAX_RETRIES = 10
RETRY_DELAY = 10  # Seconds to wait between retries
RATE_LIMIT_DELAY = 1200  # 20 minutes wait for rate limit handling

def fetch_with_retries(url, query, variables):
    retries = 0
    while retries < MAX_RETRIES:
        # Headers
        headers = {
            "Authorization": f"Bearer {get_random_token()}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(url, json={"query": query, "variables": variables}, headers=headers)
            
            if response.status_code == 200:
                return response.json(), response  # Return JSON data on success
            elif response.status_code == 403:
                print(f"Request limit reached, waiting for {RATE_LIMIT_DELAY // 60} minutes")
                time.sleep(RATE_LIMIT_DELAY)  # Wait 20 minutes before retrying
            else:
                print(f"Request failed with status code {response.status_code}: {response.text}")
                retries += 1
                print(f"Retrying {retries}/{MAX_RETRIES} after {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)  # Wait before retrying
        # Catch network-level exceptions, such as connection errors or SSL errors, and retry
        except Exception as e:
            retries += 1
            print(f"Connection error ({type(e).__name__}): {e}. Retrying {retries}/{MAX_RETRIES} after {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
    # After exceeding the maximum number of retries, write error info to log file and raise exception
    with open(f"error_{id}.log", "a") as f:
        f.write(f"Failed to fetch data after {MAX_RETRIES} retries\n")
        f.write(f"Query: {query}\n")
        f.write(f"Variables: {variables}\n")
        f.write(f"status_code: {response.status_code}\n")
        f.write(f"response: {response.text}\n")
    raise Exception("Maximum retries exceeded")

start_date = datetime(2008, 1, 1)  # GitHub was founded in 2008
end_date = datetime.now()
delta = timedelta(days=30)  # Set the time interval to 30 days; adjust as needed

 # Main loop
while start_date < end_date:
    print(f"Processing data from {start_date} to {start_date + delta}")
    curr = highThreshold         # Initialize stars limit
    while curr >= lowThreshold:
        query = get_query_with_stars_less_than_and_date_within(curr, start_date, start_date + delta)  # Get query statement
        # Complete this request chain
        while has_next_page and curr >= lowThreshold:
            variables = {"afterCursor": after_cursor}
            
            # Use the encapsulated fetch_with_retries function to get response data
            ret, response = fetch_with_retries(url, query, variables)
            data = ret.get("data", {}).get("search", {})

            # Extract repository info and write to file
            with open(output_csv, "a") as f:
                lines = []
                for repo in data["repos"]:
                    if repo['repo']['stargazerCount'] >= lowThreshold:
                        curr = min(curr, repo['repo']['stargazerCount'])
                        try:
                            git_url = convert_to_git_protocol_url(repo['repo']['url'])
                            lines.append(f"{repo['repo']['primaryLanguage']['name'] if repo['repo']['primaryLanguage'] else None},{repo['repo']['name']},{git_url},{repo['repo']['assignableUsers']['totalCount']},{repo['repo']['isFork']},{repo['repo']['isInOrganization']},{repo['repo']['stargazerCount']},{repo['repo']['createdAt']},{repo['repo']['pushedAt']},{repo['repo']['diskUsage']}\n")
                        except ValueError as e:
                            print(f"Failed to convert URL: {repo['repo']['url']}")
                total += len(lines)
                f.writelines(lines)
            
            # Update pagination control
            has_next_page = data["pageInfo"]["hasNextPage"]
            after_cursor = data["pageInfo"]["endCursor"]
            print(f"Processed with cursor {after_cursor}, request_limit: {response.headers['X-RateLimit-Remaining']}, {total} records collected")
            # If no repositories match, set curr to 0 to exit loop
            if len(data["repos"]) == 0:
                curr = 0
            curr -= 1
            
        # Replace tail recursion with loop
        has_next_page = True
        after_cursor = None
    start_date += delta

print(f"Finished processing")
