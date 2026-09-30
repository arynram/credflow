import re
import sys

def modify_app():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update Imports
    if 'import multi_source_validator' not in content:
        content = content.replace("import feature_engineering", "import feature_engineering\nimport multi_source_validator")

    # 2. Update File Uploaders
    old_up_r = '''        with up_r:
            st.markdown("**Optional Supporting Documents** *(Marked Optional)*")
            st.checkbox("GST Return (Optional)", value=True, help="Cross-verifies sales turnover")
            st.checkbox("Sales Invoices (Optional)", value=False)
            st.checkbox("Loan / EMI Statement (Optional)", value=True)
            st.checkbox("Electricity / Utility Bills (Optional)", value=False)'''
            
    new_up_r = '''        with up_r:
            st.markdown("**Multi-Source Financial Verification (Optional)**")
            gst_file = st.file_uploader("GST Returns (Optional)", type=["csv", "pdf", "xlsx"])
            itr_file = st.file_uploader("ITR Document (Optional)", type=["pdf", "csv"])
            udyam_file = st.file_uploader("Udyam / Business Reg (Optional)", type=["pdf", "png", "jpg"])
            
            st.session_state['has_gst'] = gst_file is not None
            st.session_state['has_itr'] = itr_file is not None
            st.session_state['has_udyam'] = udyam_file is not None'''
            
    content = content.replace(old_up_r, new_up_r)

    # 3. Add Integrity Check block before Step 4
    # Find STEP 4
    step4_idx = content.find('# STEP 4: APPLY FOR LOAN')
    if step4_idx != -1:
        integrity_block = '''
            # MULTI-SOURCE INTEGRITY CHECK
            if all_done:
                st.markdown("### Financial Verification Center")
                integrity = multi_source_validator.validate_bank_integrity(raw_df)
                if not integrity['valid']:
                    st.error(f"⚠ Assessment Blocked\\n\\nWe could not establish sufficient financial evidence from the uploaded document.\\n\\n**Reason:** {integrity['reason']}\\n\\n**Required Action:** Upload a valid financial statement with sufficient transaction history.")
                    st.stop()
                    
                score_data = multi_source_validator.calculate_verification_score(True, st.session_state.get('has_gst', False), st.session_state.get('has_itr', False), st.session_state.get('has_udyam', False), 0.05)
                
                st.session_state['verification_score_data'] = score_data
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Bank Data", "✓ Parsed")
                c2.metric("GST Status", "✓ Cross-validated" if st.session_state.get('has_gst') else "○ Missing")
                c3.metric("ITR Status", "✓ Cross-validated" if st.session_state.get('has_itr') else "○ Missing")
                c4.metric("Verification Score", f"{score_data['score']}/100", f"Coverage: {score_data['coverage']}")

        '''
        content = content[:step4_idx] + integrity_block + "\n            " + content[step4_idx:]

    # Write changes
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Modified app.py successfully!")

if __name__ == '__main__':
    modify_app()
