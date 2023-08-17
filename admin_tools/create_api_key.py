import datetime
from google.cloud import api_keys_v2
from google.cloud.api_keys_v2 import Key

from google.cloud import datastore
from google.cloud.datastore.query import PropertyFilter, And


def add_api_key_to_datastore(project_id: str,key: str, student_id:str,key_id:str) -> None:
    client = datastore.Client(project=project_id)
    key = client.key('ApiKey', key)
    entity = datastore.Entity(key=key)
    entity.update({
        'student_id': student_id,
        'key_id': key_id
    })
    client.put(entity)

def create_api_key(project_id: str, id:str,name: str) -> Key:
    """
    Creates and restrict an API key. Add the suffix for uniqueness.

    TODO(Developer):
    1. Before running this sample,
      set up ADC as described in https://cloud.google.com/docs/authentication/external/set-up-adc
    2. Make sure you have the necessary permission to create API keys.

    Args:
        project_id: Google Cloud project id.

    Returns:
        response: Returns the created API Key.
    """
    # Create the API Keys client.
    client = api_keys_v2.ApiKeysClient()

    key = api_keys_v2.Key()
    key.display_name = name

    # Initialize request and set arguments.
    request = api_keys_v2.CreateKeyRequest()
    request.parent = f"projects/{project_id}/locations/global"
    request.key = key
    request.key_id = id

    # Make the request and wait for the operation to complete.
    response = client.create_key(request=request).result()

    print(f"Successfully created an API key: {response.name}")
    # For authenticating with the API key, use the value in "response.key_string".
    # To restrict the usage of this API key, use the value in "response.name".
    return response

def restrict_api_key_api(project_id: str, service:str, key_id: str) -> Key:
    """
    Restricts an API key. Restrictions specify which APIs can be called using the API key.

    TODO(Developer): Replace the variables before running the sample.

    Args:
        project_id: Google Cloud project id.
        key_id: ID of the key to restrict. This ID is auto-created during key creation.
            This is different from the key string. To obtain the key_id,
            you can also use the lookup api: client.lookup_key()

    Returns:
        response: Returns the updated API Key.
    """

    # Create the API Keys client.
    client = api_keys_v2.ApiKeysClient()

    # Restrict the API key usage by specifying the target service and methods.
    # The API key can only be used to authenticate the specified methods in the service.
    api_target = api_keys_v2.ApiTarget()
    api_target.service = service
    api_target.methods = ["*"]

    # Set the API restriction(s).
    # For more information on API key restriction, see:
    # https://cloud.google.com/docs/authentication/api-keys
    restrictions = api_keys_v2.Restrictions()
    restrictions.api_targets = [api_target]

    key = api_keys_v2.Key()
    key.name = f"projects/{project_id}/locations/global/keys/{key_id}"
    key.restrictions = restrictions

    # Initialize request and set arguments.
    request = api_keys_v2.UpdateKeyRequest()
    request.key = key
    request.update_mask = "restrictions"

    # Make the request and wait for the operation to complete.
    response = client.update_key(request=request).result()

    print(f"Successfully updated the API key: {response.name}")
    # Use response.key_string to authenticate.
    return response


if __name__ == "__main__":
    project_id = "pytest-runner"
    api = "pytestrunnerapi-1l4kv4fc0t9cr.apigateway.pytest-runner.cloud.goog"
    student_id = "1234567"
    # key = create_api_key(project_id, "studentid-" + student_id ,"cywong@vtc.edu.hk")
    # print(key)   
    # response = restrict_api_key_api(project_id, api, key.uid)
    # print(response) 
    # add_api_key_to_datastore(project_id, key.key_string, student_id, key.uid)
 