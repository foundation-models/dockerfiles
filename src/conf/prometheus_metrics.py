from prometheus_client import CollectorRegistry, Counter, Gauge

from conf.cached import Cache



class PrometheusMetrics:
    def __init__(self):
        self.registry = CollectorRegistry(True)
        self.counter_exception = Counter("http_exception", "Number of exceptions events")
        self.number_of_signature = Counter("not_signature",
                                           "Number of no signatures processed")
        self.languages = Cache.spacy_languages
        self.entities = Cache.nergptconfig.entity_level_tags if Cache.nergptconfig else []
        self.gauge_score = {}
        self.number_entities = {}
        for one in self.entities:
            self.gauge_score[one] = Gauge(f"entity_score_{one}",
                                          f" Model score for {one} entity")
            self.number_entities[one] = Counter(f"entity_processed_{one}",
                                                f"Number of entity {one} processed")

        self.lang_counter = {}
        for i, one in enumerate(self.languages):
            name = one.replace("-", "")
            self.lang_counter[one] = Counter(f"lang_{name}",
                                             f"Number of lang {name} processed")

        self._initialize()


    def _initialize(self):
        self.registry.register(self.counter_exception)
        self.registry.register(self.number_of_signature)

        for element in self.entities:
            self.registry.register(self.gauge_score[element])
            self.registry.register(self.number_entities[element])

        for lg in self.languages:
            self.registry.register(self.lang_counter[lg])
