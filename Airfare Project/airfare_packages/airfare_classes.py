# should add way to change data type of columns, delete column, delete table, connect cursor to exisiting table, etc
from sqlalchemy import Table
import pandas as pd


class db_table:

    def __init__(self, table_name, db_engine, meta_data):
        self.table_name = table_name
        self.connection = None
        self.table = None
        self.db_engine = db_engine
        self.meta_data = meta_data

    def __enter__(self):
        self.connection = self.db_engine.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.connection:
            self.connection.close()

    def create_table(self, cols):

        self.table = Table(self.table_name, self.meta_data, *cols)
        self.table.create(self.db_engine, checkfirst=True)

    def initial_data_upload(self, df):

        if self.table is None:
            self.table = Table(
                self.table_name, self.meta_data, autoload_with=self.db_engine
            )

        insert_stmt = self.table.insert()
        self.connection.execute(insert_stmt, df.to_dict(orient="records"))
        self.connection.commit()

    def read_table(self):

        return pd.read_sql(f"SELECT * FROM {self.table_name}", self.db_engine)
