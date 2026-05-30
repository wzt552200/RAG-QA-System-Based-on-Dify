import arxiv

def search_arxiv(query: str, max_results=5):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results = []
    for paper in client.results(search):
        results.append({
            "title": paper.title,
            "summary": paper.summary[:500],
            "link": paper.entry_id,
            "published": paper.published.date().isoformat()
        })
    return results