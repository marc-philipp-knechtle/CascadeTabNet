"""
This python file provides access methods to the mongoDB database.
They are separated into two methods for each collection because these are the only used collections.
"""
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

# Connection defaults
MONGO_CONNECTION_DEFAULT = "mongodb://localhost:17270/?readPreference=primary&ssl=false"
COLLECTION_CASCADE_TABNET_DEFAULT = "cascadeTabNet"
DATABASE_SHARED_FILE_FORMAT_DEFAULT = "shared-file-format"


class Connection:
    __mongo_connection: str
    __collection_cascade_tab_net: str
    __database_shared_file_format: str

    def __init__(self, connection_oplog: str = MONGO_CONNECTION_DEFAULT,
                 collection_diff: str = COLLECTION_CASCADE_TABNET_DEFAULT,
                 database_local: str = DATABASE_SHARED_FILE_FORMAT_DEFAULT):
        self.__mongo_connection = connection_oplog
        self.__collection_cascade_tab_net = collection_diff
        self.__database_shared_file_format = database_local

    def get_db(self) -> Database:
        return MongoClient(self.__mongo_connection).get_database(self.__database_shared_file_format)

    def get_collection(self) -> Collection:
        return self.get_db().get_collection(self.__collection_cascade_tab_net)
