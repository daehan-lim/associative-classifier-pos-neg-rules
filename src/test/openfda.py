import requests

drug_name = "ALBUTEROL SULFATE"
url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}&limit=1"
response = requests.get(url)
data = response.json()

if data and 'results' in data:
    result = data['results'][0]

    if 'openfda' in result and 'generic_name' in result['openfda']:
        active_ingredient = result['openfda']['generic_name'][0]
        print(f"Active Ingredient: {active_ingredient}")
    if 'active_ingredient' in result:
        amount = result['active_ingredient'][0]
        print(f"Amount: {amount}")
    if 'spl_product_data_elements' in result:
        spl_product_data = result['spl_product_data_elements'][0]
        print(f"Product Data Elements: {spl_product_data}")
    else:
        print("No information found.")
else:
    print("No information found.")
