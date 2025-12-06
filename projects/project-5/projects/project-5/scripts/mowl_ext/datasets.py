import logging

from mowl.datasets import PathDataset

from util.logger_config import config

logger = logging.getLogger(__name__)
config(logger)


class OtcPathDataset(PathDataset):
    """
    Project-specific PathDataset that overrides `evaluation_classes`.

    In mOWL, Dataset.evaluation_classes must return a pair (tuple) of OWLClasses
    collections to define the candidate space used during evaluation (e.g., for
    link prediction over subclass axioms). The base class leaves this abstract.

    Proposal/default behavior:
    - Use the full set of classes gathered from training/validation/testing
      ontologies for both elements of the pair. This mirrors the typical setup
      in taxonomy link prediction where any class can be a potential head/tail.

    You can later specialize this to restrict to a subset (e.g., only classes
    appearing in simple named rdfs:subClassOf axioms) without changing callers.
    """

    @property
    def evaluation_classes(self):
        # Cache as in Dataset implementation expectations
        if self._evaluation_classes is None:
            # By default, evaluate over the complete class set for both sides
            cls = self.classes
            self._evaluation_classes = (cls, cls)
            logger.debug(
                "Initialized evaluation_classes with %d classes (pair).", len(cls)
            )
        return self._evaluation_classes
