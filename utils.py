import requests


def extract_pdf_text_via_file(file_path):
    url = "https://data.distilled.ai/pdf/extract-raw"
    headers = {"X-DISTILLED-API-KEY": "GpDQp1lYpl51Sduj4ZYg06oTkIreUZk8"}

    files = {"file": (file_path, open(file_path, "rb"), "application/pdf")}

    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        return response.json()  # Return the JSON response if successful
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")


def extract_pdf_text_via_url(file_url):
    url = "https://data.distilled.ai/pdf/extract-raw"

    headers = {"X-DISTILLED-API-KEY": "GpDQp1lYpl51Sduj4ZYg06oTkIreUZk8"}

    params = {"url": file_url}

    # Send the GET request to the API
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()  # Return the JSON response if successful
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")


def pdf_to_text(file_url, output_path):
    try:
        pdf_json = extract_pdf_text_via_url(file_url)
        result = ""
        for x in pdf_json:
            temp = x["section_name"] + "\n" + x["section_content"] + "\n"
            result += temp
        with open(output_path, "w") as f:
            f.write(result)
    except Exception as e:
        print(f"Error when extract pdf: {e}")
        return False
    return True
