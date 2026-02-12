# livertoxrag/data_loader.py
# Parses LiverTox .nxml files into text

import xml.etree.ElementTree as ET
from pathlib import Path

def load_nxml_files(folder_path: Path):
    """
    Parse LiverTox .nxml files into text, including only specific sections.
    
    Args:
        folder_path (Path): Path to folder containing .nxml files
        
    Returns:
        dict: Dictionary mapping document IDs to dictionaries of section titles and text
    """
    # Define sections to include (keep original capitalization as found in the files)
    included_sections = [
        "Introduction", "introduction",
        "Background", "background", 
        "Hepatotoxicity", "hepatotoxicity",
        "Mechanism of Injury", "mechanism of injury", "MECHANISM OF INJURY",
        "Outcome and Management", "outcome and management", "OUTCOME AND MANAGEMENT",
        "PRODUCT INFORMATION", "Product Information", "product information"
    ]
    
    docs = {}
    included_count = 0
    excluded_count = 0
    total_sections = 0
    
    for file in folder_path.glob("*.nxml"):
        try:
            tree = ET.parse(file)
            root = tree.getroot()
            sections_dict = {}  # Dictionary to store sections

            for section in root.findall(".//sec"):
                total_sections += 1
                title_elem = section.find("title")
                title = title_elem.text.strip() if title_elem is not None else "Untitled"
                
                # Check if section should be included (case-sensitive match)
                section_included = False
                
                # Direct match with our list
                for included_section in included_sections:
                    if title == included_section:
                        section_included = True
                        break
                
                # Special handling for individual case reports - include Case 1, Case 2, etc.
                if not section_included and title.startswith("Case ") and any(c.isdigit() for c in title):
                    section_included = True
                
                if not section_included:
                    excluded_count += 1
                    continue
                
                included_count += 1
                section_texts = []
                
                # Process tables if present
                tables = section.findall(".//table")
                for table in tables:
                    table_text = []
                    # Process table caption
                    caption = table.find(".//caption")
                    if caption is not None:
                        caption_text = ''.join(caption.itertext()).strip()
                        if caption_text:
                            table_text.append(f"Table Caption: {caption_text}")
                    
                    # Process table rows
                    for row in table.findall(".//tr"):
                        row_text = []
                        for cell in row.findall(".//*"):
                            if cell.tag in ["th", "td"]:
                                cell_text = ''.join(cell.itertext()).strip()
                                if cell_text:
                                    row_text.append(cell_text)
                        if row_text:
                            table_text.append(" | ".join(row_text))
                    
                    if table_text:
                        section_texts.append("\n".join(table_text))
                
                # Process paragraphs
                paras = section.findall(".//p")
                for p in paras:
                    text = ''.join(p.itertext()).strip()
                    if text:
                        section_texts.append(text)  # Store just the text
                
                if section_texts:  # Only add if there's content
                    sections_dict[title] = "\n\n".join(section_texts)
                    
            docs[file.stem] = sections_dict  # Store the dictionary of sections
        except Exception as e:
            print(f"Error parsing {file.name}: {e}")
    
    print(f"Loaded {len(docs)} documents with {included_count} included sections (excluded {excluded_count} sections)")
    print(f"Included sections: {', '.join(included_sections[:5])}... and {len(included_sections)-5} more")
    return docs