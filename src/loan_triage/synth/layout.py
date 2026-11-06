"""Printed labels for each document type, shared by the renderer and the demo engine."""

from __future__ import annotations

from loan_triage.domain.enums import DocumentType

TITLES: dict[DocumentType, str] = {
    DocumentType.LOAN_APPLICATION: "Residential Loan Application",
    DocumentType.PAYSTUB: "Earnings Statement",
    DocumentType.W2: "Wage and Tax Statement (Form W-2 Sample)",
    DocumentType.BANK_STATEMENT: "Deposit Account Statement",
    DocumentType.CREDIT_REPORT: "Consumer Credit Report",
}

LABELS: dict[DocumentType, dict[str, str]] = {
    DocumentType.LOAN_APPLICATION: {
        "applicant_name": "Borrower Name",
        "ssn_last4": "SSN (Last 4)",
        "application_date": "Application Date",
        "loan_amount": "Loan Amount Requested",
        "loan_purpose": "Loan Purpose",
        "loan_term_months": "Loan Term (Months)",
        "interest_rate": "Proposed Note Rate",
        "property_address": "Subject Property",
        "property_value": "Property Value",
        "proposed_monthly_housing": "Proposed Monthly Housing Payment",
        "stated_monthly_income": "Gross Monthly Income",
        "employer_name": "Current Employer",
        "employment_start_date": "Employment Start Date",
    },
    DocumentType.PAYSTUB: {
        "employee_name": "Employee Name",
        "employer_name": "Employer Name",
        "pay_period_end": "Pay Period Ending",
        "pay_date": "Pay Date",
        "pay_frequency": "Pay Frequency",
        "gross_pay_current": "Gross Pay",
        "total_deductions_current": "Total Deductions",
        "net_pay_current": "Net Pay",
        "gross_pay_ytd": "YTD Gross",
    },
    DocumentType.W2: {
        "employee_name": "Employee Name",
        "employer_name": "Employer Name",
        "ssn_last4": "Employee SSN",
        "tax_year": "Tax Year",
        "wages_box1": "Box 1 Wages, Tips, Other Compensation",
    },
    DocumentType.BANK_STATEMENT: {
        "account_holder_name": "Account Holder",
        "statement_period_end": "Statement Period Ending",
        "opening_balance": "Beginning Balance",
        "total_deposits": "Total Deposits",
        "total_withdrawals": "Total Withdrawals",
        "closing_balance": "Ending Balance",
        "largest_deposit": "Largest Deposit",
        "nsf_count": "NSF Items",
    },
    DocumentType.CREDIT_REPORT: {
        "consumer_name": "Consumer Name",
        "ssn_last4": "Consumer SSN",
        "report_date": "Report Date",
        "credit_score": "Credit Score",
        "total_monthly_debt_payments": "Total Monthly Debt Payments",
        "delinquencies_24m": "Delinquencies (24 Months)",
        "public_records": "Public Records",
    },
}
