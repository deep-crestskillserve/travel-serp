from locale import currency
import os
from typing import Optional
from dotenv import load_dotenv
import serpapi
from pydantic import BaseModel, Field
from langchain_core.tools import tool
_ = load_dotenv()

class FlightsInput(BaseModel):
    departure_airport: Optional[str] = Field(description='Departure airport code (IATA) or location kgmid (e.g., "CDG" or "CDG,ORY" or "/m/0vzm").')
    arrival_airport: Optional[str] = Field(description='Arrival airport code (IATA) or location kgmid (e.g., "AUS" or "LAX,SEA" or "/m/0vzm").')
    gl: Optional[str] = Field('in', description='Parameter defines the country to use for the Google Flights search. It\'s a two-letter country code. (e.g., in for India, uk for United Kingdom)')
    hl: Optional[str] = Field('en', description='Parameter defines the language to use for the Google Flights search. It\'s a two-letter language code. (e.g., en for English, es for Spanish, or fr for French).')
    currency: Optional[str] = Field('INR', description='Parameter defines the currency of the returned prices. Default to INR.')
    type: Optional[str] = Field('1', description='Flight type: "1" (Round trip, default), "2" (One-way), "3" (Multi-city).')
    outbound_date: Optional[str] = Field(description='Outbound date in YYYY-MM-DD format (e.g., "2025-08-03").')
    return_date: Optional[str] = Field(None, description='Return date in YYYY-MM-DD format (e.g., "2025-08-09"). Required for round trip, omitted for one-way.')
    travel_class: Optional[str] = Field('1', description='Travel class: "1" (Economy, default), "2" (Premium economy), "3" (Business), "4" (First).')
    adults: Optional[int] = Field(1, description='Number of adults. Default to 1.')
    children: Optional[int] = Field(0, description='Number of children. Default to 0.')
    infants_in_seat: Optional[int] = Field(0, description='Number of infants in seat. Default to 0.')
    infants_on_lap: Optional[int] = Field(0, description='Number of infants on lap. Default to 0.')
    stops: Optional[str] = Field('1', description='Number of stops: "1" (Nonstop only, default), "0" (Any stops), "2" (1 stop or fewer), "3" (2 stops or fewer).')


class FlightsInputSchema(BaseModel):
    params: FlightsInput

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
@tool(args_schema=FlightsInputSchema)
async def flights_finder(input_schema: FlightsInput):
    '''
    Find flights using the Google Flights engine.

    Args:
        input_schema: A FlightsInputSchema object containing flight search parameters.

    Returns:
        dict: Flight search results with status and response.
    '''

    input_schema = {
        'api_key': SERPAPI_API_KEY,
        'engine': 'google_flights',

        'departure_id': input_schema.departure_airport,
        'arrival_id': input_schema.arrival_airport,

        'gl': input_schema.gl,
        'hl': input_schema.hl,
        'currency': input_schema.currency,

        'type': input_schema.type,
        'outbound_date': input_schema.outbound_date,
        'return_date': input_schema.return_date,
        'travel_class': input_schema.travel_class,

        'adults': str(input_schema.adults),
        'children': str(input_schema.children),
        'infants_in_seat': str(input_schema.infants_in_seat),
        'infants_on_lap': str(input_schema.infants_on_lap),

        'stops': input_schema.stops,
    }
    params = {k: v for k, v in params.items() if v is not None}
    try:
        search = await serpapi.search(params)
        results = {
            'status': 'success',
            'response': search.data.get('best_flights', []),
            'google_flights_url': search.data.get('google_flights_url', '')
        }
    except Exception as e:
        results = {
            'status': 'error',
            'response': str(e)
        }
    return results