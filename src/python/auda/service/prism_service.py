from logging import Logger
from typing import Dict, List, Tuple
from uuid import UUID

from auda.core import Database, DatabaseName, auda
from auda.model import PrismDataExtraction, PrismDataPoint, PrismLocation
from auda.model.prism_document import PrismDocument
from sqlalchemy import func


class PrismService:
    def __init__(self):
        """Initializes a PrimService.

        Attributes:
            logger: Logger instance for logging.
            prism_db: Database instance for the PRISM database.
        """

        self.logger: Logger = auda.get_logger('PrismService')
        self.prism_db: Database = auda.database_manager.get(DatabaseName.PRISM)

    def get_data_points(self) -> List[PrismDataPoint]:
        """Fetches all data points from the Prism database.

        Returns:
            A list of PrismDataPoint objects.
        """

        with self.prism_db.get_session() as session:
            return session.query(PrismDataPoint).all()

    def get_locations(self) -> List[PrismLocation]:
        """Fetches all locations from the PRISM database.

        Returns:
            A list of PrismLocation objects.
        """

        with self.prism_db.get_session() as session:
            return session.query(PrismLocation).all()

    def get_location_map(self) -> Dict[str, PrismLocation]:
        """Returns a dictionary mapping location IDs to PrismLocation objects.

        Returns:
            A dictionary mapping location IDs to PrismLocation objects.
        """

        return {str(location.id): location for location in self.get_locations()}

    def get_data_point_map(self) -> Dict[str, PrismDataPoint]:
        """Returns a dictionary mapping data point IDs to PrismDataPoint
        objects.

        Returns:
            A dictionary mapping data point IDs to PrismDataPoint objects.
        """

        return {
            str(data_point.id): data_point
            for data_point in self.get_data_points()
        }

    def get_data_points_with_description_and_counts(
        self,
    ) -> List[Tuple[str, str, int]]:
        """Fetches data points (data columns) along with their description and
        counts (the number of data extraction records) from the PRISM database.

        Returns:
            A dictionary mapping data point names to their respective counts.
        """

        with self.prism_db.get_session() as session:
            query = (
                session.query(
                    PrismDataPoint.name,
                    PrismDataPoint.description,
                    func.count(PrismDataPoint.name).label(
                        'count_of_data_points'
                    ),
                )
                .select_from(PrismDataPoint)
                .join(
                    PrismDataExtraction,
                    PrismDataPoint.id == PrismDataExtraction.data_point_id,
                )
                .group_by(PrismDataPoint.name, PrismDataPoint.description)
                .order_by(func.count(PrismDataPoint.name).desc())
            )
            results = query.all()

            return [
                (str(name), str(description), int(count_of_data_points))
                for name, description, count_of_data_points in results
            ]

    def get_all_documents(self) -> List[PrismDocument]:
        """Fetches all documents from the PRISM database that have an
        extraction status of 'EXTRACTED'.

        Returns:
            A list of PrismDocument objects that have been extracted.
        """

        with self.prism_db.get_session() as session:
            query = (
                # How to select all columns?
                session.query(PrismDocument)
                .filter(PrismDocument.extraction_status == 'EXTRACTED')
                .order_by(PrismDocument.created_at.desc())
            )
            results = query.all()

        return list(results)

    def get_data_extraction(
        self, data_point_id: UUID, location_id: UUID, year: str, value: str
    ) -> List[PrismDataExtraction]:
        with self.prism_db.get_session() as session:
            query = session.query(PrismDataExtraction).filter(
                PrismDataExtraction.data_point_id == data_point_id,
                PrismDataExtraction.location_id == location_id,
                PrismDataExtraction.year == year,
                PrismDataExtraction.value == value,
            )
            results = query.all()

        return list(results)

    def get_document_by_id(self, document_id: UUID) -> PrismDocument:
        """Fetches a document by its ID from the PRISM database.

        Args:
            document_id (UUID): The ID of the document to fetch.
        """

        with self.prism_db.get_session() as session:
            query = session.query(PrismDocument).filter(
                PrismDocument.id == document_id
            )
            result = query.first()

        if result is None:
            raise ValueError(f'Document with ID {document_id} not found.')

        return result
