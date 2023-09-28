import datetime
import json
from os import path
import os
from pathlib import Path
import subprocess
import zipfile

import functions_framework
from google.cloud import datastore


def get_student_id_by_api_key(key: str) -> str:
    client = datastore.Client(project=os.environ.get('GCP_PROJECT'))
    student = client.get(client.key('ApiKey', key))
    return str(student['student_id'])

def save_completed_task(student_id: str, question:str, source_code:str) -> bool:
    client = datastore.Client(project=os.environ.get('GCP_PROJECT'))
    key = client.key('CompletedTask', student_id + "->" + str(question))
    entity = datastore.Entity(key=key)
    entity.update({
        'student_id': student_id,
        'question': question ,
        'source_code':source_code,
        'time': datetime.datetime.now() 
    })
    client.put(entity)


def is_marked(student_id: str, question:str) -> bool:
    client = datastore.Client(project=os.environ.get('GCP_PROJECT'))   
    task = client.get(client.key('CompletedTask', student_id + "->" + str(question)))
    return task is not None

@functions_framework.http
def pytestrunner(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    key = request_args["key"]
    print(f"key: {key}")

    student_id = get_student_id_by_api_key(key)
    print(f"student_id: {student_id}")

    if request.method == "GET":
        source_code_file_path = request_args["sourceCodeFilePath"]
        source_code = request_args["sourceCode"]
    else:
        source_code_file_path = request_json["sourceCodeFilePath"]
        source_code = request_json["sourceCode"]

    if not source_code or not source_code_file_path:
        return '{ "error": "source_code and source_code_file_path must present." }', 422
    
    print(f"source_code_file_path: {source_code_file_path}")
    print(f"source_code: \n{source_code}")

    question = source_code_file_path

    if is_marked(student_id, question):
        return 'Repeated Successful Test.', 200    

    root = path.dirname(path.abspath(__file__))

    zipped_pyTest_code = path.join(path.dirname(
    path.realpath(__file__)), 'assignments.zip')
    file = zipfile.ZipFile(zipped_pyTest_code)

    file.extractall(path=root)
    print(f"Extracted zip file to {root}")
        
    code_file_path = path.join(root, source_code_file_path)
    with open(code_file_path, 'w') as filetowrite:
        filetowrite.write(source_code)

    text = Path(code_file_path).read_text()
    print(text)
    
    # Source lab\lab01\ch01_t01_hello_world.py to Test tests\lab01\test_ch01_t01_hello_world.py
    source_code_file_path_segments = source_code_file_path.split("/")
    test_code_file_path = path.join(
        root, "tests", source_code_file_path_segments[1], "test_"+source_code_file_path_segments[2])
        
    test_result_text = os.path.join(root, 'result.json')
    
    
    cmd = f"""python -m pytest -v {test_code_file_path} --json-report --json-report-file={test_result_text}"""
    print(cmd)
    test_result = subprocess.getoutput(cmd)
    print(test_result)
    test_result_text = Path(test_result_text).read_text()
    print(test_result_text)
   
    try:
        test_result_json = json.loads(test_result_text)            
        is_all_tests_passed = test_result_json["summary"]["passed"] / \
            test_result_json["summary"]["total"] == 1
        print(f"is_all_tests_passed: {is_all_tests_passed}")

        if is_all_tests_passed:
            save_completed_task(student_id, question, source_code)             
            return "Test Success and saved the result.", 200
    
    except BaseException as ex:
        print(f"Unexpected {ex=}, {type(ex)=}")
            
    return test_result_text
