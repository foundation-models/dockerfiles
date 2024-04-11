from typing import List
from pydantic import BaseModel
from enum import Enum

class Name(BaseModel):
    wholeName: str = None
    prefix: str = None
    given: str = None
    middle: str = None
    surname: str = None
    suffix: str = None

class JobTitle(BaseModel):
    title: str = None

class Organization(BaseModel):
    name: str = None

class Address(BaseModel):
    line: str = None

class City(BaseModel):
    name: str = None

class CountryDivision(BaseModel):
    name: str = None

class PostalCode(BaseModel):
    code: str = None

class Country(BaseModel):
    name: str = None

class WholeAddress(BaseModel):
    street: str = None
    city: str = None
    countryDivision: str = None
    country: str = None
    postalCode: str = None
    
class Email(BaseModel):
    address: str = None
    tag: str = None
    wholeEmail: str = None

class PhoneNumber(BaseModel):
    number: str = None
    phoneTag: str = None
    phoneType: str = None
    extension: str = None
    extensionTag: str = None
    wholeNumber: str = None

class WebSite(BaseModel):
    url: str = None
    tag: str = None
    wholeUrl: str = None

class SignatureFields(BaseModel):
    names: List[Name] = None
    jobTitles: List[JobTitle] = None
    organizations: List[Organization] = None
    addresses: List[Address] = None
    cities: List[City] = None
    countryDivisions: List[CountryDivision] = None
    postalCodes: List[PostalCode] = None
    countries: List[Country] = None
    emails: List[Email] = None
    phoneNumbers: List[PhoneNumber] = None
    webSites: List[WebSite] = None

class SignatureFieldsNew(BaseModel):
    name: Name = None
    jobTitles: List[JobTitle] = None
    organizations: List[Organization] = None
    address: List[WholeAddress] = None
    emails: List[Email] = None
    phoneNumbers: List[PhoneNumber] = None
    webSites: List[WebSite] = None

class SignatureBlock(BaseModel):
    block: str = None

class ParseStatusType(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    NO_SIGNATURE_BLOCK = "NO_SIGNATURE_BLOCK"

class SenderEmailDataSourceTypes(str, Enum):
    SIGNATURE_EMAIL_ADDRESS = 'SIGNATURE_EMAIL_ADDRESS'
    REPLY_TO = 'REPLY_TO'
    NONE = 'NONE'
    SENDER = 'SENDER'
    FROM = 'FROM'

class SenderEmailDataEntry(BaseModel):
    address: str = None
    name: str = None
    confidenceScore: int = None
    sourceTypes: List[SenderEmailDataSourceTypes] = None

class SignatureResponse(BaseModel):
    signatureBlock: SignatureBlock = None
    parseStatusType: ParseStatusType
    signatureFields: SignatureFields = None
    senderEmailData: List[SenderEmailDataEntry] = None

class SignatureBlockResponse(BaseModel):
    body: str = None
    signature: str = None
    tail: str = None
    signature_start_index: int = None
    signature_end_index: int = None
