"""Search Jobnet.dk via its public (undocumented) BFF search endpoint.

Reverse-engineered from the jobnet.dk/find-job page's own network traffic
(see project notes) since the old job.jobnet.dk/CV/FindWork/Search API was
retired. This endpoint is NOT an official/documented API and Jobnet can
change or block it without notice -- treat failures here as expected and
degrade gracefully rather than crashing the whole search.
"""

import requests

from bot.config import HTTP_USER_AGENT
from bot.sources import Vacancy

SEARCH_URL = "https://jobnet.dk/bff/FindJob/Search"


def search(keyword: str, results_per_page: int = 50, timeout: int = 20) -> list[Vacancy]:
    resp = requests.get(
        SEARCH_URL,
        params={
            "resultsPerPage": results_per_page,
            "pageNumber": 1,
            "orderType": "BestMatch",
            "kmRadius": 50,
            "searchString": keyword,
        },
        headers={
            "x-csrf": "1",
            "Referer": "https://jobnet.dk/find-job",
            "Content-Type": "application/json",
            "User-Agent": HTTP_USER_AGENT,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    vacancies = []
    for ad in data.get("jobAds", []):
        location_parts = [
            ad.get("postalDistrictName") or "",
            ad.get("municipality") or "",
        ]
        location = " / ".join(p for p in location_parts if p)

        # Jobnet's own detail-page route: was /job/{id}, now /find-job/{id} --
        # the old path started bouncing every visitor (even logged-in ones)
        # to a MitID login wall that then 404s, regardless of whether the ad
        # is actually still live.
        url = ad.get("jobAdUrl") or f"https://jobnet.dk/find-job/{ad.get('jobAdId', '')}"

        vacancies.append(
            Vacancy(
                source="Jobnet",
                title=ad.get("title", ""),
                company=ad.get("hiringOrgName", ""),
                location=location,
                url=url,
                description=ad.get("description", ""),
            )
        )
    return vacancies
