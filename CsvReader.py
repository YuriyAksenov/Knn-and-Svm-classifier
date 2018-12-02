import os
from typing import List, Dict, Set


class Csv:
    @staticmethod
    def read_data(file_path: str, fields_names: Set, separator: str = ";", delimiter: str = ".", isHeader: bool = False):
        data: List[Set] = list()
        with open(file_path, "r") as file:
            if(isHeader):
                file.readline()
            for line in file:
                line = line.rstrip()
                data_item: Dict = dict()
                splitted_line = [element.replace(
                    delimiter, ".") for element in line.split(separator)]
                elem_index = 0
                for field in fields_names:
                    data_item[field] = splitted_line[elem_index]
                    elem_index += 1
                data.append(data_item)
        return data
