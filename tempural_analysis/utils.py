__all__ = ['write_to_excel', 'set_folder']

import os
import pandas as pd
base_directory = os.path.abspath(os.curdir)


def write_to_excel(table_writer: pd.ExcelWriter, sheet_name: str, headers: list, data: pd.DataFrame):
    """
    This function get header and data and write to excel
    :param table_writer: the ExcelWrite object
    :param sheet_name: the sheet name to write to
    :param headers: the header of the sheet
    :param data: the data to write
    :return:
    """
    workbook = table_writer.book
    if sheet_name not in table_writer.sheets:
        worksheet = workbook.add_worksheet(sheet_name)
    else:
        worksheet = workbook.get_worksheet_by_name(sheet_name)
    table_writer.sheets[sheet_name] = worksheet

    data.to_excel(table_writer, sheet_name=sheet_name, startrow=len(headers), startcol=0)
    all_format = workbook.add_format({
        'valign': 'top',
        'border': 1})
    worksheet.set_column(0, data.shape[1], None, all_format)

    # headers format
    merge_format = workbook.add_format({
        'bold': True,
        'border': 2,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True,
    })
    for i, header in enumerate(headers):
        worksheet.merge_range(first_row=i, first_col=0, last_row=i, last_col=data.shape[1], data=header,
                              cell_format=merge_format)
        # worksheet_header = pd.DataFrame(columns=[header])
        # worksheet_header.to_excel(table_writer, sheet_name=sheet_name, startrow=0+i, startcol=0)

    return


def set_folder(folder_name: str, father_folder_name: str = None, father_folder_path=None):
    """
    This function create new folder for results if does not exists
    :param folder_name: the name of the folder to create
    :param father_folder_name: the father name of the new folder
    :param father_folder_path: if pass the father folder path and not name
    :return: the new path or the father path if folder name is None
    """
    # create the father folder if not exists
    if father_folder_name is not None:
        path = os.path.join(base_directory, father_folder_name)
    else:
        path = father_folder_path
    if not os.path.exists(path):
        os.makedirs(path)
    # create the folder
    if folder_name is not None:
        path = os.path.join(path, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)

    return path
