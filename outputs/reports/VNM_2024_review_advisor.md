# Review Advisor: VNM (2024)

- Current raw score: 84.5 / 111.575
- Reference percentage: 75.73%
- Recoverable points: 27.7
- Best-case score if all review items are recovered: 112.2

## Issue Totals
- multi_select_incomplete: 1.5
- negative_score: 2.0
- null_answer: 17.2
- recoverable_zero: 7.0

## Top Actions
- Manually verify negative-score governance/social disclosure questions against source pages.
- Review query plan aliases/source filters or add goldset evidence for NULL questions.
- Review multi-select questions option-by-option.

## Highest Impact Review Items

### G.31 (G4)
- Score: -1.0 / 1.0 | Lost: 2.0
- Answer: B | Status: contradicted
- Issue: negative_score | Reason: negative_disclosure_or_non_compliance
- Sources checked: VNM_Baocaothuongnien_2024.pdf pp.72-75
- Recommendation: Xác minh wording VNSI và evidence nguồn. Nếu công ty thật sự không công bố nội dung này thì giữ điểm âm; nếu có, chỉnh query plan/source filter để kéo đúng tài liệu/trang.
- Retrieval hint: governance profile: bao cao quan tri, DHDCD, bien ban, nghi quyet, co tuc, thu lao

### S.3.2.1 (S3)
- Score: 0.0 / 1.2 | Lost: 1.2
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: VNM_Baocaothuongnien_2024.pdf pp.85-86
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: review query plan aliases + exact phrase anchors

### S.3.1.3 (S3)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: 1463573710_80d0136b9fd8926b73cc054696c389ec74fcc8845f56ac1f59c23bda4fe58d96_b5a60cb541.pdf pp.37-41
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: review query plan aliases + exact phrase anchors

### S.2.4.1 (S2)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: 1747645027_b159f24cc73223e3f20e273456b43ca4c2b8b755b0d77e4d0374f21eaeb3ae66_dbbf411dcb (1).pdf pp.89-92
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: review query plan aliases + exact phrase anchors

### S.2.3.1 (S2)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: 1680151957_5e98a010298a8600f15a004509311b5263af565d36d48c9e324dfe0b90ce780a_7f2bdba296.pdf p.5
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: review query plan aliases + exact phrase anchors

### S.2.2.6 (S2)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: VNM_Baocaothuongnien_2024.pdf pp.41-42
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: review query plan aliases + exact phrase anchors

### S.2.2.4 (S2)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: A | Status: supported
- Issue: recoverable_zero | Reason: partial_score
- Sources checked: VNM_Baocaothuongnien_2024.pdf pp.67-68
- Recommendation: Có điểm mất nhưng chưa rõ nguyên nhân; ưu tiên review query_plan, source_sections và disclosure còn thiếu.
- Retrieval hint: policy profile: chinh sach, quy trinh, he thong, chung nhan, con hieu luc

### S.2.1.2 (S2)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: 1683631993_840733d05a0ce9533cfd4eb963c14816cc15c40cd088faef088c9b32b24bd763_178620acbe.pdf pp.87-88
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: review query plan aliases + exact phrase anchors

### S.2.1.1 (S2)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: 1618816554_ee7ebf4f06d574c3578582f216943b5f34098b89872b7de19c218513271550a6_ab312b4b02.pdf pp.50-65
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: policy profile: chinh sach, quy trinh, he thong, chung nhan, con hieu luc

### S.1.4.1 (S1)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: 1620207714_e3cfc52e4ad5dff98c95899e572436272485aa1fd25939b33d90af899b376d65_48d967f7c5.pdf pp.52-53
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: policy profile: chinh sach, quy trinh, he thong, chung nhan, con hieu luc

### S.1.2.2 (S1)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: C | Status: insufficient
- Issue: recoverable_zero | Reason: evidence_missing
- Sources checked: VNMSR_Full_VN_2351c45431.pdf pp.121-122
- Recommendation: Có điểm mất nhưng chưa rõ nguyên nhân; ưu tiên review query_plan, source_sections và disclosure còn thiếu.
- Retrieval hint: policy profile: chinh sach, quy trinh, he thong, chung nhan, con hieu luc

### S.1.1.0 (S1)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: A | Status: supported
- Issue: recoverable_zero | Reason: partial_score
- Sources checked: 1747645027_b159f24cc73223e3f20e273456b43ca4c2b8b755b0d77e4d0374f21eaeb3ae66_dbbf411dcb (1).pdf pp.17-19
- Recommendation: Có điểm mất nhưng chưa rõ nguyên nhân; ưu tiên review query_plan, source_sections và disclosure còn thiếu.
- Retrieval hint: review query plan aliases + exact phrase anchors

### S.1.0. (S1)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: 1747645027_b159f24cc73223e3f20e273456b43ca4c2b8b755b0d77e4d0374f21eaeb3ae66_dbbf411dcb (1).pdf pp.86-87
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: review query plan aliases + exact phrase anchors

### G.34 (G5)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: VNM_Baocaothuongnien_2024.pdf pp.72-73
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: governance profile: bao cao quan tri, DHDCD, bien ban, nghi quyet, co tuc, thu lao

### E.3.5.3 (E3)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: 1747645027_b159f24cc73223e3f20e273456b43ca4c2b8b755b0d77e4d0374f21eaeb3ae66_dbbf411dcb (1).pdf pp.100-101
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: table profile: so lieu, ty le, tong luong, don vi, doanh thu

### E.3.5.1 (E3)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: B | Status: supported
- Issue: recoverable_zero | Reason: partial_score
- Sources checked: QES_Nh_C3_A0_m_C3_A1y_N_C6_B0_E1_BB_9_Bc_Gi_E1_BA_A3i_Kh_C3_A1t_Vi_E1_BB_87t_Nam_1f3e825bbd.pdf pp.11-22
- Recommendation: Có điểm mất nhưng chưa rõ nguyên nhân; ưu tiên review query_plan, source_sections và disclosure còn thiếu.
- Retrieval hint: review query plan aliases + exact phrase anchors

### E.3.4.1 (E3)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: VNMSR_Full_VN_2351c45431.pdf p.23
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: review query plan aliases + exact phrase anchors

### E.3.1.2 (E3)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: 1747645027_b159f24cc73223e3f20e273456b43ca4c2b8b755b0d77e4d0374f21eaeb3ae66_dbbf411dcb (1).pdf pp.99-100
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: table profile: so lieu, ty le, tong luong, don vi, doanh thu

### E.3.1.1 (E3)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: VNMSR_Full_VN_2351c45431.pdf pp.138-141
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: table profile: so lieu, ty le, tong luong, don vi, doanh thu

### E.2.2.6 (E2)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: A | Status: insufficient
- Issue: recoverable_zero | Reason: evidence_missing
- Sources checked: 1620207714_e3cfc52e4ad5dff98c95899e572436272485aa1fd25939b33d90af899b376d65_48d967f7c5.pdf pp.41-42
- Recommendation: Có điểm mất nhưng chưa rõ nguyên nhân; ưu tiên review query_plan, source_sections và disclosure còn thiếu.
- Retrieval hint: policy profile: chinh sach, quy trinh, he thong, chung nhan, con hieu luc

### E.2.2.5 (E2)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: 1747645027_b159f24cc73223e3f20e273456b43ca4c2b8b755b0d77e4d0374f21eaeb3ae66_dbbf411dcb (1).pdf p.65
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: review query plan aliases + exact phrase anchors

### E.2.2.10 (E2)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: VNM_Baocaothuongnien_2024.pdf pp.66-67
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: review query plan aliases + exact phrase anchors

### E.1.1.5 (E1)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: C | Status: insufficient
- Issue: recoverable_zero | Reason: evidence_missing
- Sources checked: 1747645027_b159f24cc73223e3f20e273456b43ca4c2b8b755b0d77e4d0374f21eaeb3ae66_dbbf411dcb (1).pdf pp.18-20
- Recommendation: Có điểm mất nhưng chưa rõ nguyên nhân; ưu tiên review query_plan, source_sections và disclosure còn thiếu.
- Retrieval hint: review query plan aliases + exact phrase anchors

### E.1.1.4 (E1)
- Score: 0.0 / 1.0 | Lost: 1.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: 1656491111_f46b334fdbaaa976e16ea1d6efa9b655f272d896982aa0e8509b8be361b1a883_87bac92f8b.pdf pp.49-50
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: policy profile: chinh sach, quy trinh, he thong, chung nhan, con hieu luc

### S.2.2.8 (S2)
- Score: 0.5 / 1.25 | Lost: 0.75
- Answer: A,D | Status: supported
- Issue: multi_select_incomplete | Reason: multi_select_incomplete
- Sources checked: 1732544594_f45e5cc953ac6d3f99e632fdff1618d3b748c32715d18718d85d1ba3224cf9e8_8d51d7635a.pdf pp.27-29
- Recommendation: Review từng option A/B/C... riêng, vì mỗi option có thể cộng điểm độc lập nếu có bằng chứng.
- Retrieval hint: extract evidence_by_option from policy/table sections

### S.2.2.1 (S2)
- Score: 0.125 / 0.875 | Lost: 0.75
- Answer: G | Status: supported
- Issue: multi_select_incomplete | Reason: multi_select_incomplete
- Sources checked: 1683631993_840733d05a0ce9533cfd4eb963c14816cc15c40cd088faef088c9b32b24bd763_178620acbe.pdf pp.47-49
- Recommendation: Review từng option A/B/C... riêng, vì mỗi option có thể cộng điểm độc lập nếu có bằng chứng.
- Retrieval hint: extract evidence_by_option from policy/table sections

### S.3.1.4 (S3)
- Score: 0.5 / 1.0 | Lost: 0.5
- Answer: A | Status: supported
- Issue: recoverable_zero | Reason: partial_score
- Sources checked: 1463573710_80d0136b9fd8926b73cc054696c389ec74fcc8845f56ac1f59c23bda4fe58d96_b5a60cb541.pdf pp.37-41
- Recommendation: Có điểm mất nhưng chưa rõ nguyên nhân; ưu tiên review query_plan, source_sections và disclosure còn thiếu.
- Retrieval hint: review query plan aliases + exact phrase anchors

### S.2.5.2 (S2)
- Score: 0.5 / 1.0 | Lost: 0.5
- Answer: C | Status: supported
- Issue: recoverable_zero | Reason: partial_score
- Sources checked: 1683631993_840733d05a0ce9533cfd4eb963c14816cc15c40cd088faef088c9b32b24bd763_178620acbe.pdf pp.31-35
- Recommendation: Có điểm mất nhưng chưa rõ nguyên nhân; ưu tiên review query_plan, source_sections và disclosure còn thiếu.
- Retrieval hint: review query plan aliases + exact phrase anchors

### E.2.3.6 (E2)
- Score: 0.0 / 0.0 | Lost: 0.0
- Answer: NULL | Status: insufficient
- Issue: null_answer | Reason: llm_null_or_no_answer
- Sources checked: 1747645027_b159f24cc73223e3f20e273456b43ca4c2b8b755b0d77e4d0374f21eaeb3ae66_dbbf411dcb (1).pdf pp.41-43
- Recommendation: Review query plan, source filters và các disclosure còn thiếu; nếu tài liệu có bằng chứng, bổ sung alias/anchor đúng vào plan hoặc goldset.
- Retrieval hint: table profile: so lieu, ty le, tong luong, don vi, doanh thu
