import json

def get_columns(detail):
    file_path = r'C:\\Projects\\Data Engineering\\Project 1 - File Format Converter\\data\\'
    file_name = 'schemas.json'
    json_file = file_path + file_name

    file = open(json_file)
    json_data = json.load(file)


    columns = [column['column_name'] for column in json_data[detail]]
    return columns

print(get_columns('products'))
