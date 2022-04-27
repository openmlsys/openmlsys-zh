import requests
import csv

def analysis_issue():
    total_data = []
    for page_num in range(20):
        print(f"Pulling page {page_num}")
        response = requests.get('https://api.github.com/repos/openmlsys/openmlsys-zh/issues', params={"state": "all", "per_page": 30, "page": page_num})
        print(f"Result status: {response}")
        data = response.json()
        if len(data) == 0:
            break
        total_data += data    

    all_issues = []
    node_ids = set([])
    for item in total_data:
        if isinstance(item, dict):
            if item["node_id"] not in node_ids:
                all_issues.append([item["user"]["login"], item["author_association"], 1, [item["url"].split("/")[-1]]])
                node_ids.add(item["node_id"])
    print(f"All issues and pr count {len(all_issues)}")
    total_count = {}
    for issue_item in all_issues:
        if total_count.get(issue_item[0], None) is None:
            total_count[issue_item[0]] = issue_item
        else:
            total_count[issue_item[0]][-2] += 1
            total_count[issue_item[0]][-1] += issue_item[-1]
    
    keys = sorted(total_count, key=lambda x: total_count[x][-2])
    
    final_res = []
    for key in keys:
        total_count[key][-1] = ",".join(total_count[key][-1])
        final_res.append(total_count[key])
        
    res_file = open("contribution_stats.csv", "w")
    csv_writer = csv.writer(res_file)
    csv_writer.writerow(["github id", "role", "issue and pr count", "issue or pr ids"])
    csv_writer.writerows(final_res)
analysis_issue()