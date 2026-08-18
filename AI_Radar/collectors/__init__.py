from .base import BaseCollector
from .huggingface import HuggingFaceCollector
from .github_repos import GitHubReposCollector
from .vendor_blogs import VendorBlogsCollector
from .arxiv import ArxivCollector
from .openrouter import OpenRouterCollector
from .reuters import ReutersCollector
from .theinformation import TheInformationCollector
from .semianalysis import SemiAnalysisCollector

ALL_COLLECTORS = [
    HuggingFaceCollector,
    GitHubReposCollector,
    VendorBlogsCollector,
    ArxivCollector,
    OpenRouterCollector,
    ReutersCollector,
    TheInformationCollector,
    SemiAnalysisCollector,
]
