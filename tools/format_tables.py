from bs4 import BeautifulSoup
import os

root_path = "./"
html_root_path = "_build/html/"

def get_html_list():
    index_html_path = os.path.join(root_path,html_root_path ,"index.html")
    index_soup = BeautifulSoup(open(index_html_path))

    content_list = index_soup.find(name="div", attrs={"class":"globaltoc"}).\
        find_all(name="a", attrs={"class": "reference internal"})
    html_list = [os.path.join(html_root_path,content_name["href"]) for content_name in content_list]
    return html_list

def format_table():
    html_list = get_html_list()
    for html_file in html_list:
        soup = BeautifulSoup(open(html_file))
        all_tables = soup.find_all(name="table", attrs={"class":"docutils align-default"})
        for table in all_tables:
            table["style"] = "margin:auto"
        if len(all_tables):
            write_out_file = open(html_file, mode="w")
            write_out_file.write(soup.prettify())
            write_out_file.close()
            
if __name__ == "__main__":
    format_table()