def extract_epsg(crs_info_str):
    try:
        # Search for EPSG code in the crs_info string
        epsg_code_start = crs_info_str.find('ID["EPSG".6')
        epsg_code_end = crs_info_str.find(']]', epsg_code_start)
        epsg_code = crs_info_str[epsg_code_start:epsg_code_end].split("'")[2].strip()
        return epsg_code
    except Exception as e:
        print(f"Error extracting EPSG code: {e}, {crs_info_str}")
        return '28193'

