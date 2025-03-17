"""
Clickhouse module for ann-benchmarks
"""
import clickhouse_connect
import subprocess
import sys
import os

from typing import Dict, Any, Optional

from ..base.module import BaseANN
from ...util import get_bool_env_var


class clickhouse(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']

    def fit(self, X):
        self._chclient = clickhouse_connect.get_client()
        self._chclient.query('DROP TABLE IF EXISTS items')
        print("Fitting")
        self._chclient.query('CREATE TABLE items (id Int32, vector Array(Float32)) ENGINE=MergeTree ORDER BY (id)')
        rows = []
        for i, embedding in enumerate(X):
            row = [i, embedding]
            rows.append(row)
        self._chclient.insert('items', rows, column_names=['id', 'vector'])
        print("Optimizing")
        self._chclient.query('OPTIMIZE TABLE items FINAL SETTINGS mutations_sync=2')
        print("Adding Index")
        if self._metric == "angular":
            distfn = "cosineDistance"
        else:
            distfn = "L2Distance"
        self._chclient.query('SET allow_experimental_vector_similarity_index=1')
        add_index = "ALTER TABLE items ADD INDEX vector_index vector TYPE vector_similarity('hnsw','" + distfn + "', 'f32', " + str(self._m) + "," + str(self._ef_construction) + ")"
        print(add_index)
        self._chclient.query(add_index)
        print("Building Index")
        # TODO - add receive_timeout else below comand errors out after 300s
        self._chclient.query('ALTER TABLE items MATERIALIZE INDEX vector_index SETTINGS mutations_sync = 2')
        return

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search

    def query(self, v, n):
        params = {'v1':list(v), 'v2':n}
        ef_search_str = " SETTINGS hnsw_candidate_list_size_for_search=" + str(self._ef_search)
        if self._metric == "angular":
            result = self._chclient.query("SELECT id FROM items ORDER BY cosineDistance(vector, %(v1)s) LIMIT %(v2)s" + ef_search_str,
                                      parameters=params)
        else:
            result = self._chclient.query("SELECT id FROM items ORDER BY L2Distance(vector, %(v1)s) LIMIT %(v2)s" + ef_search_str,
                                      parameters=params)
        rows = []
        for f in result.result_rows:
            rows.append(f[0])
        return rows

    def get_memory_usage(self):
        return 0

    def __str__(self):
        return f"clickhouse(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
