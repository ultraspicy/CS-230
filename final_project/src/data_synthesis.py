import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_transactions(num_transactions=1000, start_date="2024-01-01"):
    # Define category hierarchies
    category_mapping = {
        # Food & Dining
        "Groceries": "Food & Dining",
        "Restaurants": "Food & Dining",
        "Coffee Shops": "Food & Dining",
        "Fast Food": "Food & Dining",
        "Food Delivery": "Food & Dining",
        "Bars & Nightlife": "Food & Dining",
        "Bakeries": "Food & Dining",
        "Specialty Food": "Food & Dining",
        "Meal Kit Services": "Food & Dining",
        "Food Trucks": "Food & Dining",
        "International Grocery": "Food & Dining",
        "Farmers Markets": "Food & Dining",
        "Wine & Liquor": "Food & Dining",
        "Bubble Tea": "Food & Dining",
        "Ice Cream & Desserts": "Food & Dining",
        "Vending Machines": "Food & Dining",
        "Campus Dining": "Food & Dining",
        "Food Court": "Food & Dining",
        "Catering Services": "Food & Dining",
        "Smoothie & Juice Bars": "Food & Dining",
        
        # Transportation
        "Car & Gas": "Transportation",
        "Public Transit": "Transportation",
        "Ride Share": "Transportation",
        "Parking": "Transportation",
        "Tolls": "Transportation",
        "Car Wash": "Transportation",
        "Auto Parts": "Transportation",
        "Auto Service": "Transportation",
        "Auto Insurance": "Transportation",
        "Motorcycle Expenses": "Transportation",
        "Electric Vehicle Charging": "Transportation",
        "Bicycle Maintenance": "Transportation",
        "Car Registration": "Transportation",
        "Traffic Tickets": "Transportation",
        "Motorcycle Gear": "Transportation",
        "Motorcycle Parts": "Transportation",
        "Motorcycle Insurance": "Transportation",
        "Scooter Rentals": "Transportation",
        "Bike Share": "Transportation",
        "Ferry": "Transportation",
        "Train": "Transportation",
        "Bridge Tolls": "Transportation",
        "Highway Tolls": "Transportation",
        "Airport Parking": "Transportation",
        "Street Parking": "Transportation",
        "Garage Parking": "Transportation",
        "Car Detailing": "Transportation",
        "Tire Services": "Transportation",
        "Oil Change": "Transportation",
        "Car Battery": "Transportation",
        "Car Accessories": "Transportation",
        "Car Paint & Body": "Transportation",
        "Car Rental Insurance": "Transportation",
        "Driver License Fees": "Transportation",
        "Vehicle Inspection": "Transportation",
        "Roadside Assistance": "Transportation",
        "Car Loan Payment": "Transportation",
        
        # Financial Services
        "Insurance": "Financial Services",
        "Investment Fees": "Financial Services",
        "Bank Fees": "Financial Services",
        "Financial Advisor": "Financial Services",
        "Tax Preparation": "Financial Services",
        "Life Insurance": "Financial Services",
        "Property Insurance": "Financial Services",
        "Trading Commissions": "Financial Services",
        "Credit Card Fees": "Financial Services",
        "Retirement Account Fees": "Financial Services",
        "Legal Services": "Financial Services",
        "Accounting Services": "Financial Services",
        "Cryptocurrency Fees": "Financial Services",
        "Wire Transfer Fees": "Financial Services",
        "Foreign Transaction Fees": "Financial Services",
        "Overdraft Fees": "Financial Services",
        "Safe Deposit Box": "Financial Services",
        "Notary Services": "Financial Services",
        "Credit Report Fees": "Financial Services",
        "Identity Protection": "Financial Services",
        "Estate Planning": "Financial Services",
        "Financial Software": "Financial Services",
        
        # Housing
        "Rent": "Housing",
        "Mortgage": "Housing",
        "HOA Fees": "Housing",
        "Property Tax": "Housing",
        "Home Insurance": "Housing",
        "Home Repairs": "Housing",
        "Home Improvement": "Housing",
        "Lawn Care": "Housing",
        "Pest Control": "Housing",
        "Security System": "Housing",
        "Moving Expenses": "Housing",
        "Storage": "Housing",
        "Furniture": "Housing",
        "Home Decor": "Housing",
        "Cleaning Services": "Housing",
        "Smart Home Devices": "Housing",
        "Window Treatments": "Housing",
        "Appliance Repair": "Housing",
        "HVAC Service": "Housing",
        "Plumbing": "Housing",
        "Electrical": "Housing",
        "Roofing": "Housing",
        "Flooring": "Housing",
        "Painting": "Housing",
        "Landscaping": "Housing",
        "Pool Maintenance": "Housing",
        "Home Inspection": "Housing",
        "Property Management": "Housing",
        "Renters Insurance": "Housing",
        "Air Filters": "Housing",
        "Light Fixtures": "Housing",
        "Home Warranty": "Housing",
        "Locksmith": "Housing",
        "Garage Door Service": "Housing",
        
        # Technology & Electronics
        "Computer Hardware": "Technology & Electronics",
        "Software Subscriptions": "Technology & Electronics",
        "Mobile Apps": "Technology & Electronics",
        "Cloud Storage": "Technology & Electronics",
        "Gaming Services": "Technology & Electronics",
        "Tech Support": "Technology & Electronics",
        "Electronics Repair": "Technology & Electronics",
        "Phone Cases": "Technology & Electronics",
        "Screen Protectors": "Technology & Electronics",
        "Phone Accessories": "Technology & Electronics",
        "Computer Accessories": "Technology & Electronics",
        "Printer Supplies": "Technology & Electronics",
        "Camera Equipment": "Technology & Electronics",
        "Audio Equipment": "Technology & Electronics",
        "Smart Watches": "Technology & Electronics",
        "Tablets": "Technology & Electronics",
        "E-readers": "Technology & Electronics",
        "Gaming Consoles": "Technology & Electronics",
        "Gaming Accessories": "Technology & Electronics",
        "VR Equipment": "Technology & Electronics",
        "Drones": "Technology & Electronics",
        "Smart Home Hubs": "Technology & Electronics",
        "Network Equipment": "Technology & Electronics",
        "Data Recovery": "Technology & Electronics",
        "Computer Upgrades": "Technology & Electronics",
        
        # Entertainment & Streaming
        "Netflix": "Entertainment & Streaming",
        "Hulu": "Entertainment & Streaming",
        "Disney+": "Entertainment & Streaming",
        "HBO Max": "Entertainment & Streaming",
        "Amazon Prime Video": "Entertainment & Streaming",
        "YouTube Premium": "Entertainment & Streaming",
        "Spotify": "Entertainment & Streaming",
        "Apple Music": "Entertainment & Streaming",
        "Twitch": "Entertainment & Streaming",
        "Game Pass": "Entertainment & Streaming",
        "PlayStation Plus": "Entertainment & Streaming",
        "Nintendo Online": "Entertainment & Streaming",
        "Movie Tickets": "Entertainment & Streaming",
        "Concert Tickets": "Entertainment & Streaming",
        "Sports Events": "Entertainment & Streaming",
        "Theater Tickets": "Entertainment & Streaming",
        "Museum Passes": "Entertainment & Streaming",
        "Audiobook Services": "Entertainment & Streaming",
        "Podcast Subscriptions": "Entertainment & Streaming",
        "Live Streaming": "Entertainment & Streaming",
        
        # Healthcare
        "Primary Care": "Healthcare",
        "Specialist Care": "Healthcare",
        "Dental": "Healthcare",
        "Vision": "Healthcare",
        "Pharmacy": "Healthcare",
        "Mental Health": "Healthcare",
        "Physical Therapy": "Healthcare",
        "Chiropractic": "Healthcare",
        "Lab Tests": "Healthcare",
        "Medical Equipment": "Healthcare",
        "Health Insurance": "Healthcare",
        "Alternative Medicine": "Healthcare",
        "Urgent Care": "Healthcare",
        "Orthodontics": "Healthcare",
        "Dermatology": "Healthcare",
        "Cardiology": "Healthcare",
        "Neurology": "Healthcare",
        "Pediatrics": "Healthcare",
        "OB/GYN": "Healthcare",
        "Radiology": "Healthcare",
        "Vaccination": "Healthcare",
        "Medical Records": "Healthcare",
        "Ambulance": "Healthcare",
        "Hospital Stays": "Healthcare",
        "Medical Transportation": "Healthcare",
        "Telehealth": "Healthcare",
        "Medical Insurance Copay": "Healthcare",
        "Prescription Delivery": "Healthcare",
        "Contact Lenses": "Healthcare",
        "Eyeglasses": "Healthcare",
        "Dental Insurance": "Healthcare",
        "Vision Insurance": "Healthcare",
        
        # Education & Career
        "Tuition": "Education & Career",
        "Textbooks": "Education & Career",
        "School Supplies": "Education & Career",
        "Online Courses": "Education & Career",
        "Professional Training": "Education & Career",
        "Certifications": "Education & Career",
        "Educational Software": "Education & Career",
        "Professional Memberships": "Education & Career",
        "Student Loans": "Education & Career",
        "Test Prep": "Education & Career",
        "Tutoring": "Education & Career",
        "Language Learning": "Education & Career",
        "Professional Development": "Education & Career",
        "Career Coaching": "Education & Career",
        "Resume Services": "Education & Career",
        "Job Search": "Education & Career",
        "Work Uniforms": "Education & Career",
        "Work Equipment": "Education & Career",
        "Research Materials": "Education & Career",
        "Lab Fees": "Education & Career",
        "Study Abroad": "Education & Career",
        "Educational Travel": "Education & Career",
        "Professional License": "Education & Career",
        "Interview Prep": "Education & Career",
        "Coding Bootcamp": "Education & Career",
        "Technical Training": "Education & Career",
        "Industry Conferences": "Education & Career",
        "Workshop Fees": "Education & Career",
        "Educational Apps": "Education & Career",
        "Academic Publications": "Education & Career",
        
        # Utilities & Communications
        "Electricity": "Utilities & Communications",
        "Water": "Utilities & Communications",
        "Gas": "Utilities & Communications",
        "Garbage": "Utilities & Communications",
        "Phone": "Utilities & Communications",
        "Internet": "Utilities & Communications",
        "Cable TV": "Utilities & Communications",
        "Mobile Data": "Utilities & Communications",
        "VPN Service": "Utilities & Communications",
        "Domain Names": "Utilities & Communications",
        "Web Hosting": "Utilities & Communications",
        "Email Service": "Utilities & Communications",
        "Landline": "Utilities & Communications",
        "International Calling": "Utilities & Communications",
        "Satellite TV": "Utilities & Communications",
        "Solar Power": "Utilities & Communications",
        "Smart Meter": "Utilities & Communications",
        "Home Security Monitoring": "Utilities & Communications",
        
        # Personal Care & Beauty
        "Hair Care": "Personal Care & Beauty",
        "Spa Services": "Personal Care & Beauty",
        "Cosmetics": "Personal Care & Beauty",
        "Nail Care": "Personal Care & Beauty",
        "Personal Hygiene": "Personal Care & Beauty",
        "Skin Care": "Personal Care & Beauty",
        "Barber": "Personal Care & Beauty",
        "Hair Coloring": "Personal Care & Beauty",
        "Hair Products": "Personal Care & Beauty",
        "Makeup": "Personal Care & Beauty",
        "Beauty Tools": "Personal Care & Beauty",
        "Perfume": "Personal Care & Beauty",
        "Waxing": "Personal Care & Beauty",
        "Tanning": "Personal Care & Beauty",
        "Teeth Whitening": "Personal Care & Beauty",
        "Beauty Subscription Boxes": "Personal Care & Beauty",
        "Massage": "Personal Care & Beauty",
        "Facials": "Personal Care & Beauty",
        "Laser Treatment": "Personal Care & Beauty",
        "Dermatology Cosmetic": "Personal Care & Beauty",
        
        # Hobbies & Recreation
        "Photography": "Hobbies & Recreation",
        "Art Supplies": "Hobbies & Recreation",
        "Musical Instruments": "Hobbies & Recreation",
        "Craft Supplies": "Hobbies & Recreation",
        "Sports Equipment": "Hobbies & Recreation",
        "Camping Gear": "Hobbies & Recreation",
        "Fishing Equipment": "Hobbies & Recreation",
        "Hunting Gear": "Hobbies & Recreation",
        "Board Games": "Hobbies & Recreation",
        "Collectibles": "Hobbies & Recreation",
        "Model Building": "Hobbies & Recreation",
        "Garden Supplies": "Hobbies & Recreation",
        "Sewing & Knitting": "Hobbies & Recreation",
        "Woodworking": "Hobbies & Recreation",
        "Rock Climbing": "Hobbies & Recreation",
        "Skiing": "Hobbies & Recreation",
        "Surfing": "Hobbies & Recreation",
        "Dancing": "Hobbies & Recreation",
        "Pottery": "Hobbies & Recreation",
        "Cooking Classes": "Hobbies & Recreation",
        
        # Travel & Vacations
        "Flights": "Travel & Vacations",
        "Hotels": "Travel & Vacations",
        "Vacation Rentals": "Travel & Vacations",
        "Car Rentals": "Travel & Vacations",
        "Travel Insurance": "Travel & Vacations",
        "Cruises": "Travel & Vacations",
        "Tours": "Travel & Vacations",
        "Travel Gear": "Travel & Vacations",
        "Passport Fees": "Travel & Vacations",
        "Visa Fees": "Travel & Vacations",
        "Airport Lounges": "Travel & Vacations",
        "Travel Vaccinations": "Travel & Vacations",
        "Currency Exchange": "Travel & Vacations",
        "International Phone": "Travel & Vacations",
        "Travel Guides": "Travel & Vacations",
        "Baggage Fees": "Travel & Vacations",
        "Travel Toiletries": "Travel & Vacations",
        "Travel Clothing": "Travel & Vacations",
        "Souvenirs": "Travel & Vacations",
        "Travel Photography": "Travel & Vacations",
        
        # Investments & Trading
        "Stocks": "Investments & Trading",
        "Bonds": "Investments & Trading",
        "Mutual Funds": "Investments & Trading",
        "ETFs": "Investments & Trading",
        "Cryptocurrency": "Investments & Trading",
        "Options Trading": "Investments & Trading",
        "Forex Trading": "Investments & Trading",
        "Real Estate Investment": "Investments & Trading",
        "Precious Metals": "Investments & Trading",
        "Trading Platform Fees": "Investments & Trading",
        "Investment Research": "Investments & Trading",
        "Portfolio Management": "Investments & Trading",
        "Robo-Advisor Fees": "Investments & Trading",
        "Investment Newsletter": "Investments & Trading",
        "Market Data": "Investments & Trading",
        "Trading Education": "Investments & Trading",

        # Fitness & Wellness
        "Gym Membership": "Fitness & Wellness",
        "Personal Training": "Fitness & Wellness",
        "Fitness Classes": "Fitness & Wellness",
        "CrossFit": "Fitness & Wellness",
        "Yoga Classes": "Fitness & Wellness",
        "Pilates": "Fitness & Wellness",
        "Swimming Pool": "Fitness & Wellness",
        "Sports League Fees": "Fitness & Wellness",
        "Marathon Registration": "Fitness & Wellness",
        "Fitness Equipment": "Fitness & Wellness",
        "Fitness Clothing": "Fitness & Wellness",
        "Fitness Trackers": "Fitness & Wellness",
        "Nutritionist": "Fitness & Wellness",
        "Diet Programs": "Fitness & Wellness",
        "Health Coaching": "Fitness & Wellness",
        "Meditation Apps": "Fitness & Wellness",
        "Wellness Retreats": "Fitness & Wellness",
        "Sports Medicine": "Fitness & Wellness",
        "Recovery Services": "Fitness & Wellness",
        "Health Food Supplements": "Fitness & Wellness",
        "Protein Supplements": "Fitness & Wellness",
        "Vitamins": "Fitness & Wellness",
        "Athletic Tape": "Fitness & Wellness",
        "Sports Drinks": "Fitness & Wellness",
        
        # Professional Services
        "Legal Consultation": "Professional Services",
        "Tax Services": "Professional Services",
        "Financial Planning": "Professional Services",
        "Real Estate Agent": "Professional Services",
        "Interior Design": "Professional Services",
        "Architecture Services": "Professional Services",
        "Wedding Planning": "Professional Services",
        "Career Counseling": "Professional Services",
        "Life Coaching": "Professional Services",
        "Business Consulting": "Professional Services",
        "Marketing Services": "Professional Services",
        "Web Design": "Professional Services",
        "Graphic Design": "Professional Services",
        "Photography Services": "Professional Services",
        "Videography": "Professional Services",
        "Translation Services": "Professional Services",
        "Notary Services": "Professional Services",
        "Document Preparation": "Professional Services",
        "Professional Writing": "Professional Services",
        "IT Consulting": "Professional Services",
        
        # Child & Family
        "Childcare": "Child & Family",
        "Babysitting": "Child & Family",
        "Daycare": "Child & Family",
        "After School Programs": "Child & Family",
        "Children's Activities": "Child & Family",
        "Children's Sports": "Child & Family",
        "Children's Music Lessons": "Child & Family",
        "Children's Art Classes": "Child & Family",
        "Children's Clothing": "Child & Family",
        "Children's Books": "Child & Family",
        "Children's Toys": "Child & Family",
        "Children's Furniture": "Child & Family",
        "Baby Supplies": "Child & Family",
        "Diapers": "Child & Family",
        "Baby Food": "Child & Family",
        "Baby Healthcare": "Child & Family",
        "Family Photography": "Child & Family",
        "Family Counseling": "Child & Family",
        "Family Vacations": "Child & Family",
        "Parent Education": "Child & Family",
        
        # Home Office
        "Office Furniture": "Home Office",
        "Office Supplies": "Home Office",
        "Printer": "Home Office",
        "Ink & Toner": "Home Office",
        "Paper Products": "Home Office",
        "Office Electronics": "Home Office",
        "Desk Accessories": "Home Office",
        "Filing Systems": "Home Office",
        "Office Lighting": "Home Office",
        "Office Decor": "Home Office",
        "Ergonomic Equipment": "Home Office",
        "Video Conference Equipment": "Home Office",
        "Office Software": "Home Office",
        "Office Phone": "Home Office",
        "Office Internet": "Home Office",
        "Office Security": "Home Office",
        
        # Digital Services & Software
        "Cloud Storage": "Digital Services & Software",
        "Password Managers": "Digital Services & Software",
        "Antivirus Software": "Digital Services & Software",
        "Project Management Tools": "Digital Services & Software",
        "Design Software": "Digital Services & Software",
        "Video Editing Software": "Digital Services & Software",
        "Audio Software": "Digital Services & Software",
        "Development Tools": "Digital Services & Software",
        "Business Software": "Digital Services & Software",
        "Accounting Software": "Digital Services & Software",
        "Tax Software": "Digital Services & Software",
        "Analytics Tools": "Digital Services & Software",
        "Website Builders": "Digital Services & Software",
        "Domain Registration": "Digital Services & Software",
        "Email Services": "Digital Services & Software",
        "Cloud Computing": "Digital Services & Software",
        "Digital Storage": "Digital Services & Software",
        "Digital Security": "Digital Services & Software",
        
        # Automotive Specialty
        "Performance Parts": "Automotive Specialty",
        "Custom Wheels": "Automotive Specialty",
        "Car Audio": "Automotive Specialty",
        "Window Tinting": "Automotive Specialty",
        "Car Wraps": "Automotive Specialty",
        "Engine Modifications": "Automotive Specialty",
        "Suspension": "Automotive Specialty",
        "Transmission Service": "Automotive Specialty",
        "Brake Service": "Automotive Specialty",
        "Exhaust System": "Automotive Specialty",
        "Car Electronics": "Automotive Specialty",
        "Car Security Systems": "Automotive Specialty",
        "Paint Protection": "Automotive Specialty",
        "Ceramic Coating": "Automotive Specialty",
        "Performance Tuning": "Automotive Specialty",
        "Racing Equipment": "Automotive Specialty",
        
        # Motorcycle Specialty
        "Motorcycle Gear": "Motorcycle Specialty",
        "Motorcycle Parts": "Motorcycle Specialty",
        "Motorcycle Accessories": "Motorcycle Specialty",
        "Motorcycle Tools": "Motorcycle Specialty",
        "Motorcycle Racing": "Motorcycle Specialty",
        "Motorcycle Training": "Motorcycle Specialty",
        "Motorcycle Storage": "Motorcycle Specialty",
        "Motorcycle Transport": "Motorcycle Specialty",
        "Track Days": "Motorcycle Specialty",
        "Motorcycle Clubs": "Motorcycle Specialty",
        "Motorcycle Events": "Motorcycle Specialty",
        "Motorcycle Insurance": "Motorcycle Specialty",
        "Motorcycle Maintenance": "Motorcycle Specialty",
        "Motorcycle Modifications": "Motorcycle Specialty",
        "Motorcycle Safety Gear": "Motorcycle Specialty",
        
        # Smart Home
        "Smart Lighting": "Smart Home",
        "Smart Thermostats": "Smart Home",
        "Smart Security": "Smart Home",
        "Smart Doorbells": "Smart Home",
        "Smart Locks": "Smart Home",
        "Smart Speakers": "Smart Home",
        "Smart Displays": "Smart Home",
        "Smart Appliances": "Smart Home",
        "Smart Outlets": "Smart Home",
        "Smart Sensors": "Smart Home",
        "Home Automation": "Smart Home",
        "Smart Irrigation": "Smart Home",
        "Smart Garage": "Smart Home",
        "Smart Entertainment": "Smart Home",
        "Smart Home Hub": "Smart Home",
        "Smart Home Installation": "Smart Home",
        
        # Crypto & Digital Assets
        "Bitcoin": "Crypto & Digital Assets",
        "Ethereum": "Crypto & Digital Assets",
        "NFTs": "Crypto & Digital Assets",
        "Mining Equipment": "Crypto & Digital Assets",
        "Crypto Trading Fees": "Crypto & Digital Assets",
        "Crypto Wallet": "Crypto & Digital Assets",
        "Blockchain Services": "Crypto & Digital Assets",
        "Crypto Research": "Crypto & Digital Assets",
        "Crypto Events": "Crypto & Digital Assets",
        "Digital Asset Storage": "Crypto & Digital Assets",
        "Crypto Education": "Crypto & Digital Assets",
        "Mining Software": "Crypto & Digital Assets",
        "Crypto Security": "Crypto & Digital Assets",
        
        # Pet Specialty
        "Pet Grooming": "Pet Specialty",
        "Pet Training": "Pet Specialty",
        "Pet Daycare": "Pet Specialty",
        "Pet Boarding": "Pet Specialty",
        "Pet Photography": "Pet Specialty",
        "Pet Insurance": "Pet Specialty",
        "Pet Dental Care": "Pet Specialty",
        "Pet Medicine": "Pet Specialty",
        "Pet Emergency Care": "Pet Specialty",
        "Pet Accessories": "Pet Specialty",
        "Pet Travel": "Pet Specialty",
        "Pet Food Delivery": "Pet Specialty",
        "Pet DNA Testing": "Pet Specialty",
        "Pet Microchipping": "Pet Specialty",
        "Pet Adoption Fees": "Pet Specialty",
        "Pet Memorial": "Pet Specialty"
    }
    
    # Common merchants for each category
    merchants = {
        # Food & Dining
        "Groceries": ["Whole Foods", "Trader Joe's", "Safeway", "Costco", "Target", "Walmart", "Kroger", "Albertsons", "Aldi", "Publix", "H Mart", "99 Ranch Market", "Sprouts", "Smart & Final", "Food Lion"],
        "Restaurants": ["Chipotle", "McDonald's", "Subway", "Shake Shack", "The Cheesecake Factory", "Olive Garden", "Red Lobster", "P.F. Chang's", "Applebee's", "Buffalo Wild Wings"],
        "Coffee Shops": ["Starbucks", "Dunkin'", "Peet's Coffee", "Blue Bottle", "Philz Coffee", "Dutch Bros", "Coffee Bean & Tea Leaf", "Tim Hortons", "Local Coffee Shop"],
        "Fast Food": ["McDonald's", "Burger King", "Wendy's", "Taco Bell", "KFC", "Popeyes", "In-N-Out", "Five Guys", "Chick-fil-A", "Jack in the Box"],
        
        # Transportation
        "Car & Gas": ["Shell", "Chevron", "ExxonMobil", "BP", "Costco Gas", "Sam's Club Gas", "76", "Arco", "Marathon", "Speedway"],
        "Ride Share": ["Uber", "Lyft", "Via", "Gett", "Juno", "RideAustin", "Arcade City", "HopSkipDrive"],
        "Public Transit": ["MTA", "BART", "Metro", "Clipper", "Caltrain", "Amtrak", "NJ Transit", "WMATA", "CTA", "SEPTA"],
        
        # Financial Services
        "Insurance": ["State Farm", "Geico", "Progressive", "Allstate", "Liberty Mutual", "Farmers", "Nationwide", "USAA", "MetLife", "Prudential"],
        "Investment Fees": ["Fidelity", "Charles Schwab", "Vanguard", "E*TRADE", "TD Ameritrade", "Robinhood", "Merrill Lynch", "Morgan Stanley"],
        
        # Housing
        "Rent": ["Equity Residential", "AvalonBay", "Essex Property", "UDR", "Camden Property", "MAA", "Greystar", "Related Companies", "Apartment Payment"],
        "Home Improvement": ["Home Depot", "Lowe's", "Menards", "Ace Hardware", "True Value", "Harbor Freight", "Sherwin-Williams", "Floor & Decor"],
        
        # Technology & Electronics
        "Computer Hardware": ["Apple", "Best Buy", "Microsoft Store", "Dell", "HP", "Lenovo", "ASUS", "Micro Center", "B&H Photo", "Newegg"],
        "Software Subscriptions": ["Microsoft 365", "Adobe Creative Cloud", "Dropbox", "Google Workspace", "Zoom", "Slack", "AutoCAD", "Norton"],
        
        # Entertainment & Streaming
        "Streaming Services": ["Netflix", "Hulu", "Disney+", "HBO Max", "Amazon Prime Video", "YouTube Premium", "Apple TV+", "Paramount+", "Peacock", "Discovery+"],
        "Gaming Services": ["Steam", "PlayStation Network", "Xbox Live", "Nintendo eShop", "Epic Games", "EA Play", "Ubisoft+", "Google Play Games"],
        
        # Healthcare
        "Pharmacy": ["CVS", "Walgreens", "Rite Aid", "Walmart Pharmacy", "Costco Pharmacy", "Sam's Club Pharmacy", "Target Pharmacy", "Express Scripts"],
        "Health Insurance": ["UnitedHealth", "Anthem", "Aetna", "Cigna", "Humana", "Kaiser Permanente", "Blue Cross Blue Shield", "Oscar Health"],
        
        # Utilities & Communications
        "Phone": ["Verizon", "AT&T", "T-Mobile", "Sprint", "Google Fi", "Mint Mobile", "Cricket Wireless", "Metro by T-Mobile", "Boost Mobile"],
        "Internet": ["Comcast Xfinity", "Spectrum", "Cox", "AT&T Internet", "Verizon Fios", "CenturyLink", "Frontier", "Google Fiber", "WOW!"],
        
        # Travel & Vacations
        "Airlines": ["United Airlines", "American Airlines", "Delta", "Southwest", "JetBlue", "Alaska Airlines", "Spirit", "Frontier", "Hawaiian Airlines"],
        "Hotels": ["Marriott", "Hilton", "Hyatt", "IHG", "Wyndham", "Choice Hotels", "Airbnb", "VRBO", "Best Western", "La Quinta"],
        
        # Fitness & Wellness
        "Gym Membership": ["24 Hour Fitness", "LA Fitness", "Planet Fitness", "Equinox", "Crunch Fitness", "Gold's Gym", "Anytime Fitness", "YMCA"],
        "Fitness Apps": ["Peloton", "Nike Training Club", "MyFitnessPal Premium", "Strava", "Fitbod", "BeachBody", "Apple Fitness+", "FitOn"],
        
        # Smart Home
        "Smart Home": ["Ring", "Nest", "Philips Hue", "Arlo", "Ecobee", "August", "Eufy", "Wyze", "SimpliSafe", "Sonos"],
        
        # Professional Services
        "Legal Services": ["LegalZoom", "Rocket Lawyer", "Avvo", "Legal Shield", "Local Law Firm", "Notary Service", "DocuSign"],
        "Tax Services": ["H&R Block", "TurboTax", "Jackson Hewitt", "Liberty Tax", "TaxAct", "Tax Preparation Service"]
    }

    # Generate transactions
    transactions = []
    date = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Add recurring transactions
    recurring_dates = []
    for _ in range(num_transactions):
        category = np.random.choice(list(category_mapping.keys()))
        parent_category = category_mapping[category]
        
        # Select merchant based on category
        if category in merchants:
            description = np.random.choice(merchants[category])
        else:
            description = f"Payment for {category}"
        
        # Generate realistic amounts based on category
        if category == "Rent":
            amount = np.random.uniform(1500, 4000)
        elif category == "Groceries":
            amount = np.random.uniform(20, 300)
        elif category == "Restaurants":
            amount = np.random.uniform(10, 200)
        elif category == "Car & Gas":
            amount = np.random.uniform(30, 100)
        elif category == "Insurance":
            amount = np.random.uniform(50, 300)
        elif "Travel" in category:
            amount = np.random.uniform(200, 1000)
        else:
            amount = np.random.uniform(10, 500)
            
        # Determine if it should be excluded (internal transfers, etc.)
        is_transfer = False
        if np.random.random() < 0.1:  # 10% chance of being a transfer
            is_transfer = True
            if "Payment" in description or "Transfer" in description:
                amount = np.round(amount, 2)
            description = np.random.choice([
                "Payment To Chase Card",
                "Transfer to Savings",
                "Zelle Payment",
                "Venmo Transfer",
                "ACH Transfer",
                "Direct Deposit",
                "Investment Transfer"
            ])
            
        # Generate account information
        account_types = ["Checking", "Savings", "Credit Card"]
        account = np.random.choice(account_types)
        account_mask = str(np.random.randint(1000, 9999))
        
        # Add transaction
        transaction = {
            "date": date.strftime("%Y-%m-%d"),
            "description": description,
            "amount": round(amount, 2),
            "status": np.random.choice(["pending", "posted"]),
            "category": "" if is_transfer else category,
            "parent_category": "" if is_transfer else parent_category,
            "excluded": is_transfer,
            "type": "internal transfer" if is_transfer else "regular",
            "account": account,
            "account_mask": account_mask
        }
        
        transactions.append(transaction)
        date -= timedelta(days=np.random.randint(0, 3))
    
    df = pd.DataFrame(transactions)
    return df

# Generate sample data
transactions_df = generate_synthetic_transactions(100000)

# Sort by date
transactions_df = transactions_df.sort_values('date', ascending=False)

# Display first few rows
print(transactions_df.head().to_string())

# Save to CSV
transactions_df.to_csv("./../resources/synthetic_transactions.csv", index=False)