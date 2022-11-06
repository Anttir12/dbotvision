import logging

import analyzer
from dbot_api_client import DbotApiClient

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')


logger = logging.getLogger(__name__)


def try_create_api_client():
    try:
        dbot_api_client = DbotApiClient()
    except Exception:  # noqa
        dbot_api_client = None
    return dbot_api_client


def main():
    logger.info("Starting")
    dbot_api_client = try_create_api_client()
    kf_analyzer = analyzer.KillFeedAnalyzer(dbot_api_client, print_killfeed=True,
                                            act_instantly=True, combo_cutoff=3,
                                            show_debug_img=False, debug=True)
    kf_analyzer.start_analyzer()


if __name__ == "__main__":
    main()
