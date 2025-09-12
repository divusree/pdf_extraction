import streamlit as st
import zipfile
import io
from PIL import Image
import base64
from extraction import *
import json

def main():
    st.set_page_config(
        page_title="Construction Document Processor",
        page_icon="",
        layout="wide"
    )
    
    st.title("Construction Document Processor")
    st.markdown("*AI-powered extraction from Construction Drawings & Specifications*")
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("📄 Document Upload")
        
        # Document type dropdown
        doc_type = st.selectbox(
            "Document Type",
            ["Construction Drawings", "Specifications"],
            help="Select the type of construction document you're uploading"
        )
        
        # File upload
        uploaded_file = st.file_uploader(
            f"Upload {doc_type} PDF",
            type=['pdf'],
            help="Upload your construction document in PDF format"
        )
        
        # Page range selection
        st.header("📖 Page Range")
        col1, col2 = st.columns(2)
        
        with col1:
            start_page = st.number_input(
                "Start Page",
                min_value=1,
                value=1,
                step=1,
                help="Starting page number (1-indexed)"
            )
        
        with col2:
            end_page = st.number_input(
                "End Page",
                min_value=1,
                value=1,
                step=1,
                help="Ending page number (1-indexed)"
            )
        
        # Validation
        if start_page > end_page:
            st.error("Start page must be less than or equal to end page")
        
        # Process button
        process_btn = st.button(
            "🔍 Process Document",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True
        )
    
    # Main content area
    if uploaded_file is not None:
        st.success(f"✅ {doc_type} PDF uploaded successfully!")
        
        # Document info
        st.info(f"📋 **Document Type:** {doc_type} | **Pages to Process:** {start_page} to {end_page}")
        
        if process_btn:
            with st.spinner(f"Processing {doc_type.lower()} from pages {start_page} to {end_page}..."):
                # Placeholder for actual processing logic
                # In a real implementation, you would:
                # 1. Extract pages from the PDF
                # 2. Perform OCR and data extraction
                # 3. Generate structured data
                # 4. Create images for display
                page_numbers = list(range(start_page, end_page+1))
                pdf_bytes = io.BytesIO(uploaded_file.getvalue())
                is_sheet = False
                if doc_type == "Specifications":
                    laparams=LAParams(all_texts=True, detect_vertical=False, 
                                        line_overlap=0.5, char_margin=2000.0, 
                                        line_margin=0.5, word_margin=2,
                                        boxes_flow=1)    
                    layout, images = extract_everything_from_pdf(pdf_bytes, page_numbers = page_numbers, laparams=laparams)  
                    scope_misc_match, toc_pages = collect_section_info(layout)
                else:
                    laparams=LAParams(all_texts=True, detect_vertical=False, 
                                            line_overlap=0.1, char_margin=0.2, 
                                            line_margin=0.1, word_margin=0.2,
                                            boxes_flow=1)  
                    layout, images = extract_everything_from_pdf(pdf_bytes, page_numbers = page_numbers, laparams=laparams)  
                    is_sheet = True
                results = {}
                for num in page_numbers:
                    results[num] = process_page(layout, num, images, is_sheet = is_sheet)

                st.success("✅ Document processing completed!")
                
                # Display results section
                st.header("📊 Extraction Results")
                
                # Create tabs for different views
                tab1, tab2 = st.tabs(["🖼️ Visual Results", "📋 Structured Data"] ) #, "📥 Download"])
                
                with tab1:
                    st.subheader("Extracted Images")
                    
                    # Display 2 PIL images side by side
                    col1, col2 = st.columns(2)
                    # print(results[page_numbers[0]]['image'].size)
                    with col1:
                        st.markdown("**Image 1: Extracted Bboxes**")
                        # Placeholder image - replace with actual PIL image
                        temp_img = results[page_numbers[0]]['image'].copy()
                        temp_img.thumbnail((800,600))
                        # st.image(temp_img, caption="Processed floor plan with annotations", width = 'stretch')
                        tabs = st.tabs([f"Image {i}" for i in page_numbers])
                        for i, tab in enumerate(tabs):
                            with tab:
                                temp_img = results[page_numbers[i]]['image'].copy()
                                temp_img.thumbnail((800,600))                                
                                st.image(temp_img, caption=f"Detailed view of Image {i+1}", width = 'stretch')                    

                with tab2:
                    st.subheader("Structured Data Extract")
                    st.json({k:v['table_regions'] for k, v in results.items()})
                    
                    # Sample structured data based on document type
                    if doc_type == "Specifications":
                        st.json(dict(scope_misc_match))
                        st.json(toc_pages)
                
                # with tab3:
                #     st.subheader("Download Processed Results")
                #     st.markdown("Download a ZIP file containing all extracted data and images.")
                    
                #     # Create download button
                #     zip_buffer = create_zipfile(doc_type, results)
                    
                #     st.download_button(
                #         label="📥 Download ZIP File",
                #         data=zip_buffer.getvalue(),
                #         file_name=f"{doc_type.lower().replace(' ', '_')}_extraction_results.zip",
                #         mime="application/zip",
                #         type="primary",
                #         use_container_width=True
                #     )
                    
                #     st.info("💡 ZIP file contains: extracted images, structured data (JSON), processing logs, and metadata.")
    
    else:
        # Welcome message when no file is uploaded
        st.markdown("""
        ### 🚀 Welcome to Construction Document Processor
        
        This application helps you extract structured information from construction documents:
        
        **📋 Supported Document Types:**
        - **Construction Drawings**: Floor plans, elevations, details, MEP drawings
        - **Specifications**: CSI format specifications, material requirements, standards
        
        **🔧 Features:**
        - Page-range selection for targeted extraction
        - Visual analysis and annotation
        - Structured data output (JSON format)
        - Downloadable results package
        - Cross-referencing between plans and specs
        
        **📁 To get started:** Upload a PDF document using the sidebar panel.
        """)
        
        # Sample workflow diagram
        st.markdown("---")
        st.subheader("🔄 Processing Workflow")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **1. Upload & Configure**
            - Select document type
            - Choose page range
            - Upload PDF file
            """)
        
        with col2:
            st.markdown("""
            **2. AI Processing**
            - OCR text extraction
            - Image segmentation
            - Data structuring
            """)
        
        # with col3:
        #     st.markdown("""
        #     **3. Results & Export**
        #     - Visual annotations
        #     - Structured JSON data
        #     - ZIP download package
        #     """)


def create_zipfile(doc_type, results):
    """Create a sample ZIP file with extraction results"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
    
        image_buffers = [io.BytesIO()] * len(results)
        json_data = {}
        buff_ct = 0
        for page_num, page_results in results.items():
            if page_results.get("image"):
                page_results.get("image").save(image_buffers[buff_ct], format='PNG')
                zip_file.writestr(f"{page_num}_analysis.png", image_buffers[buff_ct].getvalue())
                buff_ct += 1
            else:
                json_data[page_num] = page_results
        # Add structured data as JSON
        zip_file.writestr(f"{doc_type.lower().replace(' ', '_')}_data.json", json_data)
        
    zip_buffer.seek(0)
    return zip_buffer

if __name__ == "__main__":
    main()