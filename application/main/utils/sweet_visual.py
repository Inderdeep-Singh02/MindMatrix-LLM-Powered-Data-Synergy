import sweetviz as sv
from bs4 import BeautifulSoup

async def sweet_visual(df):
    report = sv.analyze(df)
    report_html_path = "sweetviz_report.html"
    report.show_html(report_html_path,open_browser=False)

    with open(report_html_path, "r", encoding="utf-8") as report_file:
        soup = BeautifulSoup(report_file, "html.parser")
        logo_container = soup.find("div", class_="pos-logo-group")
        if logo_container:
            logo_container.decompose()

    with open(report_html_path, "w", encoding="utf-8") as modified_report_file:
        modified_report_file.write(str(soup))

    with open(report_html_path, "r", encoding="utf-8") as modified_report_file:
        modified_report_html = modified_report_file.read()

    return modified_report_html
