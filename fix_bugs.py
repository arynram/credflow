import re

def fix_app_bugs():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # --- Fix 1: The reset bug ---
    old_upload_logic = """            if cust_file is not None:
                st.session_state['has_uploaded_statement'] = True
                st.session_state['confirmed_cands'] = {}  # reset for new file
                st.session_state['parsed_statement_result'] = None  # fresh AI parse
                st.session_state['owner_submitted'] = False"""
                
    new_upload_logic = """            if cust_file is not None:
                if st.session_state.get('last_uploaded_filename') != cust_file.name:
                    st.session_state['last_uploaded_filename'] = cust_file.name
                    st.session_state['has_uploaded_statement'] = True
                    st.session_state['confirmed_cands'] = {}  # reset for new file
                    st.session_state['parsed_statement_result'] = None  # fresh AI parse
                    st.session_state['owner_submitted'] = False"""
                    
    content = content.replace(old_upload_logic, new_upload_logic)
    
    # Indent the parsing lines under the new if block
    old_parsing = """                if cust_file.name.endswith(".csv"):
                    try:
                        st.session_state['owner_df'] = pd.read_csv(cust_file)
                    except UnicodeDecodeError:
                        st.session_state['owner_df'] = pd.read_csv(cust_file, encoding='latin1')
                else:
                    st.session_state['owner_df'] = pd.read_excel(cust_file)
                st.success(f"Uploaded: {cust_file.name} successfully parsed.")"""
                
    new_parsing = """                    if cust_file.name.endswith(".csv"):
                        try:
                            st.session_state['owner_df'] = pd.read_csv(cust_file)
                        except UnicodeDecodeError:
                            st.session_state['owner_df'] = pd.read_csv(cust_file, encoding='latin1')
                    else:
                        st.session_state['owner_df'] = pd.read_excel(cust_file)
                # Show success regardless of if it's newly uploaded or already in state
                st.success(f"Uploaded: {cust_file.name} successfully parsed.")"""
                
    content = content.replace(old_parsing, new_parsing)


    # --- Fix 2: all_done NameError ---
    # We will remove the injected MULTI-SOURCE INTEGRITY CHECK and put it AFTER all_done is defined.
    
    old_integrity = """            
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

        
            # STEP 4: APPLY FOR LOAN & FINANCIAL SUMMARY (REAL DYNAMIC METRICS FROM USER'S FILE!)
            all_done = (completed_count == total_cands) and (total_cands > 0)
            if all_done:"""
            
    new_integrity = """            # STEP 4: APPLY FOR LOAN & FINANCIAL SUMMARY (REAL DYNAMIC METRICS FROM USER'S FILE!)
            all_done = (completed_count == total_cands) and (total_cands > 0)
            if all_done:
                
                # MULTI-SOURCE INTEGRITY CHECK
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
                
                st.markdown("---")
"""
    
    content = content.replace(old_integrity, new_integrity)

    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Fixed bugs successfully!")

if __name__ == '__main__':
    fix_app_bugs()
