#!/usr/bin/env python3
"""
Extract social network from edgifoia MongoDB using spaCy NER.
Optimized for speed and quality with better filtering.
"""

from pymongo import MongoClient
import spacy
from collections import defaultdict
from itertools import combinations
import json
import re

# Load spaCy model with only NER enabled for speed
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
nlp.max_length = 15000

# Patterns to filter out (email artifacts, etc.)
BAD_PATTERNS = [
    r'@',  # Email addresses
    r'\.gov',
    r'\.com',
    r'\.org',
    r'\.edu',
    r'<|>',  # Email brackets
    r'Cc:|To:|From:|Subject:|Re:',  # Email headers
    r'EXCHANGELABS|Exchange Administrative',
    r'/OU[=:]',
    r'^\d+$',  # Pure numbers
    r'^[A-Z]{2,}$',  # All caps acronyms (keep longer mixed case)
    r'Office of',  # Generic office names
    r'AGENCY$',
    r'TRUMP$',  # Misidentified as org
    r'\bSent\b',  # Email sent artifacts
    r'\bAve\b|\bBlvd\b|\bSt\b|\bRd\b',  # Street addresses
    r'====|----|\*\*\*',  # OCR garbage
    r'\bL\s*$',  # Trailing L from email signatures
    r"'s$",  # Possessives (Scott Pruitt's)
    r'^Admin\s',  # Admin prefix
    r"^cc'd",  # cc'd prefix
    r'\bMiliari\b',  # OCR garbage
    r'\bHuppj\b',  # OCR garbage
    r'\bMiilan\b',  # OCR typo
    r'\bMl\b',  # OCR garbage
    r'\+',  # Plus sign (Pruitt + Meeting)
    r'\s-\s',  # Dashes with spaces (Pruitt - April)
    r'^Ms\s',  # Ms prefix
    r'^Mr\s',  # Mr prefix
    r'\bLyndsay\b.*\bHey\b|\bHey\b.*\bLyndsay\b',  # Email greeting fragments
    r'^N\.W\.',  # Address prefix (N.W. Washington)
    r'\bSuite\b',  # Address word
    r'\bFloor\b',  # Address word
    r'^SW\s',  # SW address prefix
    r'^NW\s',  # NW address prefix
    r'\.pdf$',  # File extensions
    r'\bAdministrator\b',  # Title in entity
    r'^Dear\s',  # Letter greeting
    r'^CO\s',  # CC/CO prefix
    r'\d{1,2}$',  # Trailing numbers (Millan Hupp 4)
    r'^MILLAN\s.*\sTeam\b',  # OCR garbage patterns
    r'ford\.ha',  # Truncated email addresses
    r'Huppf|Millau|Bolog',  # OCR typos
    r'William Jefferson Clinton.*(North|Building|Phone)',  # Building name fragments
    r'\b(Visits|Touching|Looping|Morocco|Pencil|Briefing)\b',  # Email fragments after names
    r'^M\.\s',  # Single initial prefix (M. Hupp)
    r'\.pdf\b',  # PDF fragments
    r'^Red Level\b',  # Red Level prefix
    r'\bNorth\b.*\b(ford|Phone|WJC)\b',  # Address/building fragments
    r'Syndey\b',  # Typo of Sydney
    r'\br\.e\b',  # Email artifact
    r'\bBuilding\b|\bBldg\b',  # Building names
    r'^mailto\b',  # mailto artifacts
    r'rso n\b',  # OCR garbage
    r'\bCs"?\?J\.',  # OCR garbage
    r'\bTravel\b',  # Travel in name
    r'\bInvitation\b',  # Email fragments
    r'\bRemind\b',  # Email fragments
    r'\bIssues\b',  # Email fragments
    r'^Sec\s',  # Secretary abbreviation
    r'\s&\s',  # Ampersand combinations (Sydney & Aaron)
    r'\sJeff$',  # Trailing Jeff
    r'\*',  # Asterisk
    r'^N\.W$',  # Just N.W
    r'[a-f0-9]{20,}',  # Long hex strings (hashes)
    r'^RECI\s',  # OCR garbage
    r'U-Turn',  # Not a valid entity
    r'Huppi\b',  # OCR typo
    r'hupp\.\w+',  # Email address fragments
    r'^Agency[A-Z]',  # Merged words
    r'^Headquarters\s',  # Location prefix
    r'Diplomate\b',  # Restaurant name
    r'\bAudience\b',  # Email word
    r'\bCentral\b',  # Email word
    r'\bTransmission\b',  # Utility company
    r'\bKey$',  # Trailing key
    r'\bMarriott?\b',  # Hotel names
    r'\bHilton\b',  # Hotel names
    r'\bHotel\b',  # Hotel word
    r'\bContinues\b',  # News headline word
    r'\bState Action\b',  # News phrase
    r'\bTour\b',  # Tour word
    r'\bCAUTION\b',  # Email warning
    r'\bQuestions\b',  # Email word
    r'\bMight\b',  # Email word
    r'\bDon$',  # Trailing Don
    r'/Annual',  # Calendar artifact
    r'\bImailto\b',  # Email artifact
    r'\bDate$',  # Trailing Date
    r'\bPayee\b',  # Financial document
    r'^\w+\^\^',  # OCR garbage with carets
    # Corporate suffixes and patterns
    r'\b(Inc|Corp|LLC|Ltd|Co|Company|Partners|Associates|Group|Holdings|International|Industries|Services|Solutions|Consulting|Technologies|Enterprises|Foundation|Institute|Association|Corporation|Incorporated)\.?$',
    r'\bLLP\b',  # Law firm designation
    r'\bL\.L\.P\.?\b',  # Law firm designation variant
    r'^Jones Day$',  # Law firm
    r'^Baker Botts',  # Law firm
    r'^BAKER BOTTS',  # Law firm
    r'^Faegre Baker',  # Law firm
    # Known corporations that look like person names
    r'^Baker Hughes$',
    r'^General Electric$',
    r'^General Motors$',
    r'^Koch Industries$',
    r'^Dow Chemical$',
    r'^DuPont$',
    r'^ExxonMobil$',
    r'^Chevron$',
    r'^ConocoPhillips$',
    r'^Phillips 66$',
    r'^Marathon Oil$',
    r'^Halliburton$',
    r'^Schlumberger$',
    r'^Peabody Energy$',
    r'^Arch Coal$',
    r'^Murray Energy$',
    r'^Alpha Natural$',
    r'^American Petroleum$',
    r'^American Chemistry$',
    r'^American Gas$',
    r'^National Mining$',
    r'^Western Energy$',
    r'^Southern Company$',
    r'^Duke Energy$',
    r'^Dominion Energy$',
    r'^FirstEnergy$',
    r'^American Electric$'
    r'\.docx?$',  # Word document extensions
    r'\bForm\b',  # Form word
    r'\bRequest\b',  # Request word
    r'^Q&A$',  # Q&A
    r'\bMeets\b',  # Meeting word
    r'\bLetter\b',  # Letter word
    r'\bCal\b',  # Calendar
    r'\bAcct\b',  # Account
    r'Yes-$',  # Email artifact
    r'^Team\s',  # Team prefix
    r'\bParticipation\b',  # Generic word
    r'\bAddresses\b',  # Generic word
    r'^Honorable\s',  # Title prefix
    r'\s(to|are|is|was|were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|might|must|shall|can|the|a|an)$',  # Function words at end
]

# Common single-word email response fragments that often get attached to names
RESPONSE_WORDS = {
    'great', 'fantastic', 'completely', 'sorry', 'shale', 'michelle', 'jessica',
    'perfect', 'wonderful', 'excellent', 'absolutely', 'definitely', 'certainly',
    'ok', 'okay', 'sure', 'yes', 'no', 'maybe', 'probably', 'possibly',
    'regarding', 'concerning', 'about', 'per', 'via', 'for', 'with',
    # More email fragments
    'visits', 'visiting', 'touching', 'looping', 'morocco', 'pencil',
    'briefing', 'building', 'north', 'south', 'east', 'west', 'phone',
    'level', 'red', 'pdf', 'ex', 'syndey',  # typo of sydney
    'r.e', 're', 'fwd', 'fw',
    # Even more fragments
    'issues', 'invitation', 'remind', 'lauren', 'land', 'fitzgerald',
    'administrations', 'blair', 'forrest', 'mcmurray', 'sec',
}

# Words that indicate an entity is actually an email fragment
# These commonly appear after a person's name in email text
EMAIL_FRAGMENT_WORDS = {
    'subject', 'begin', 'hi', 'hey', 'hello', 'good', 'thanks', 'thank',
    'sounds', 'visit', 'speaking', 'meeting', 'call', 'importance',
    'external', 'april', 'may', 'june', 'july', 'august', 'september',
    'october', 'november', 'december', 'january', 'february', 'march',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'advance', 'office', 'executive', 'personal', 'email', 'fri', 'mon', 'tue',
    'wed', 'thu', 'sat', 'sun', 'pm', 'am', 'edt', 'est', 'pst', 'cst',
    'u', 's', 'environmental',  # "Scott Pruitt U. S. Environmental"
}

# Known person names that often get misclassified as ORGs
# Map them to canonical person names
KNOWN_PERSONS = {
    'sydney hupp': 'Sydney Hupp',
    'millan hupp': 'Millan Hupp',
    'milan hupp': 'Millan Hupp',  # typo variation
    'scott pruitt': 'Scott Pruitt',
    'pruitt, scott': 'Scott Pruitt',  # comma format
    'ryan jackson': 'Ryan Jackson',
    'william jefferson clinton': 'Bill Clinton',
    'william j. clinton': 'Bill Clinton',
    'william j clinton': 'Bill Clinton',
    'jefferson clinton': 'Bill Clinton',
}

# Known surnames - used to detect "Lastname Word" patterns that are invalid
KNOWN_SURNAMES = {
    'pruitt', 'hupp', 'jackson', 'clinton', 'trump', 'obama', 'bush',
    'jefferson', 'william', 'ford', 'beck', 'scott', 'zinke', 'dravis',
}

# Known org name mappings for fusion
ORG_ALIASES = {
    'epa': 'EPA',
    'EPA': 'EPA',
    'Environmental Protection Agency': 'EPA',
    'U.S. Environmental Protection Agency': 'EPA',
    'Sierra Club': 'Sierra Club',
    'SIERRA CLUB': 'Sierra Club',
    'Congress': 'Congress',
    'Senate': 'Senate',
    'House': 'House',
}


def normalize_name(name):
    """Normalize entity name."""
    name = ' '.join(name.split())
    # Remove titles
    name = re.sub(r'^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+', '', name, flags=re.IGNORECASE)
    # Remove leading/trailing punctuation
    name = name.strip('.,;:!?()[]{}<>=/')
    return name.strip()


def is_bad_entity(name, entity_type='PERSON'):
    """Check if entity should be filtered out."""
    if len(name) < 3:
        return True

    words = name.split()

    # For PERSON entities, restrict to 2-3 words only
    if entity_type == 'PERSON':
        if len(words) > 3:
            return True
        if len(words) < 2:
            return True

    # Filter patterns like "J.W." "J.W" etc. - these are usually not real names
    for word in words:
        # Single letter followed by dot or single letter words
        if re.match(r'^[A-Z]\.$', word) or re.match(r'^[A-Z]$', word):
            # Allow one initial at the start, but not multiple or in the middle
            if words.index(word) > 0 or sum(1 for w in words if re.match(r'^[A-Z]\.?$', w)) > 1:
                return True
        # J.W. pattern (two initials)
        if re.match(r'^[A-Z]\.[A-Z]\.?$', word):
            return True

    for pattern in BAD_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return True
    # Filter out if contains multiple > or < (email artifact)
    if name.count('>') > 0 or name.count('<') > 0:
        return True
    # Filter if looks like email header (word followed by colon)
    if re.match(r'^[A-Za-z]+:', name):
        return True
    # Filter if contains @
    if '@' in name:
        return True
    # Filter duplicated first word (e.g., "Sydney Sydney Hupp")
    words = name.split()
    if len(words) >= 2 and words[0].lower() == words[1].lower():
        return True
    # Filter OCR garbage (special chars)
    if re.search(r'[=~\[\]{}|\\]', name):
        return True
    # Filter if contains email fragment words (e.g., "Pruitt Speaking Good")
    words_lower = [w.lower().rstrip('.,;:') for w in words]
    for word in words_lower:
        if word in EMAIL_FRAGMENT_WORDS:
            return True
    # Filter if starts with common non-name words
    if words_lower and words_lower[0] in {'the', 'a', 'an', 'this', 'that', 'cc', 'mr', 'ms', 'mrs', 'dr'}:
        return True

    # Filter "Lastname ResponseWord" patterns (e.g., "Pruitt Great", "Pruitt Sorry")
    if len(words) == 2:
        first_lower = words_lower[0]
        second_lower = words_lower[1]

        # If first word is a known surname and second is a response word, filter it
        if first_lower in KNOWN_SURNAMES and second_lower in RESPONSE_WORDS:
            return True

        # If first word is a known surname and second word is a common first name
        # This catches "Pruitt Sydney", "Pruitt Michelle", etc.
        common_first_names = {'sydney', 'ryan', 'scott', 'michael', 'david', 'john', 'james',
                             'robert', 'william', 'richard', 'joseph', 'thomas', 'mary', 'jennifer',
                             'linda', 'elizabeth', 'barbara', 'susan', 'jessica', 'sarah', 'karen',
                             'millan', 'hayley', 'michelle', 'aaron', 'cheryl', 'aurelia', 'nancy',
                             'donald', 'barack', 'george', 'hillary', 'joe', 'kamala'}
        if first_lower in KNOWN_SURNAMES and second_lower in common_first_names:
            return True

    return False


def get_canonical_name(name, name_type):
    """Get canonical form of a name for matching.

    For PERSON: reduces to "firstname lastname" only, stripping all middle names/initials.
    Examples:
        "E. Scott Pruitt" -> "scott pruitt"
        "Ryan T Jackson" -> "ryan jackson"
        "Pruitt, Scott" -> "scott pruitt"
        "John Michael Smith Jr." -> "john smith"
    """
    name = normalize_name(name)

    # Check org aliases first
    if name_type == 'ORG' and name in ORG_ALIASES:
        return ORG_ALIASES[name].lower()

    if name_type == 'PERSON':
        parts = name.split()
        if len(parts) >= 2:
            # Handle "Last, First" format
            if parts[0].endswith(','):
                first = parts[1] if len(parts) > 1 else parts[0].rstrip(',')
                last = parts[0].rstrip(',')
            else:
                # Filter out single-letter initials and suffixes
                filtered_parts = []
                for p in parts:
                    clean = p.rstrip('.,')
                    # Skip single letters (initials)
                    if len(clean) == 1:
                        continue
                    # Skip common suffixes
                    if clean.lower() in {'jr', 'sr', 'ii', 'iii', 'iv', 'esq', 'phd', 'md'}:
                        continue
                    filtered_parts.append(clean)

                if len(filtered_parts) >= 2:
                    first = filtered_parts[0]
                    last = filtered_parts[-1]
                elif len(filtered_parts) == 1:
                    # Only one part left, can't make a proper name
                    return name.lower()
                else:
                    first = parts[0].rstrip('.,')
                    last = parts[-1].rstrip('.,')

            # Remove any trailing punctuation
            last = last.rstrip('.,')
            first = first.rstrip('.,')

            canonical = f"{first} {last}"
            return canonical.lower()

    return name.lower()


class EntityFuser:
    """Handles entity deduplication and fusion."""

    def __init__(self):
        self.canonical_to_names = defaultdict(set)
        self.name_to_canonical = {}
        self.entity_types = {}
        self.canonical_counts = defaultdict(int)

    def add_entity(self, name, entity_type):
        """Add an entity, fusing with existing if similar."""
        name = normalize_name(name)
        if not name or len(name) < 2 or is_bad_entity(name, entity_type):
            return None

        # Check if this is a known person (fixes ORG misclassification)
        name_lower = name.lower()
        if name_lower in KNOWN_PERSONS:
            name = KNOWN_PERSONS[name_lower]
            entity_type = 'PERSON'

        canonical = get_canonical_name(name, entity_type)

        # Also check canonical form against known persons
        if canonical in KNOWN_PERSONS:
            name = KNOWN_PERSONS[canonical]
            entity_type = 'PERSON'
            canonical = get_canonical_name(name, entity_type)

        if canonical in self.canonical_to_names:
            self.canonical_to_names[canonical].add(name)
            self.name_to_canonical[name] = canonical
            self.canonical_counts[canonical] += 1
            return self.get_best_name(canonical)

        self.canonical_to_names[canonical].add(name)
        self.name_to_canonical[name] = canonical
        self.entity_types[canonical] = entity_type
        self.canonical_counts[canonical] = 1
        return name

    def get_best_name(self, canonical):
        """Get the best (most readable) name for a canonical form."""
        names = self.canonical_to_names[canonical]
        # Prefer: title case, 2-word names (first last), no initials
        def score(n):
            s = 0
            parts = n.split()

            # Strong preference for 2-word names (First Last)
            if len(parts) == 2:
                s += 100

            # Penalize all caps
            if n.isupper():
                s -= 50

            # Prefer title case
            if n.istitle():
                s += 30

            # Penalize single-letter parts (initials)
            for p in parts:
                if len(p.rstrip('.')) == 1:
                    s -= 20

            # Small bonus for moderate length
            if 8 <= len(n) <= 20:
                s += 10

            return s
        return max(names, key=score)


def extract_entities_from_spacy_doc(spacy_doc, fuser):
    """Extract PERSON and ORG entities from a processed spaCy doc."""
    people = set()
    orgs = set()

    for ent in spacy_doc.ents:
        name = normalize_name(ent.text)

        if ent.label_ == 'PERSON':
            if is_bad_entity(name, 'PERSON'):
                continue
            if len(name.split()) >= 2:
                fused_name = fuser.add_entity(name, 'PERSON')
                if fused_name:
                    people.add(fused_name)
        elif ent.label_ == 'ORG':
            if is_bad_entity(name, 'ORG'):
                continue
            fused_name = fuser.add_entity(name, 'ORG')
            if fused_name:
                orgs.add(fused_name)

    return people, orgs


def build_cooccurrence_graph(db, batch_size=500, progress_interval=2000):
    """Build a co-occurrence graph from MongoDB documents using batch processing."""
    fuser = EntityFuser()
    nodes = defaultdict(lambda: {'count': 0, 'docs': [], 'type': 'person'})
    edges = defaultdict(lambda: {'weight': 0, 'doc_ids': set()})

    docs_collection = db.documents
    total_docs = docs_collection.count_documents({})
    print(f"Processing {total_docs} documents from edgifoia (batch size: {batch_size})...")

    person_count = 0
    org_count = 0
    processed = 0

    batch_docs = []
    batch_texts = []

    for doc in docs_collection.find():
        text = doc.get('text', '')[:10000]
        if text:
            batch_docs.append(doc)
            batch_texts.append(text)

        if len(batch_docs) >= batch_size:
            for spacy_doc, orig_doc in zip(nlp.pipe(batch_texts, batch_size=batch_size), batch_docs):
                doc_id = str(orig_doc.get('_id'))
                hash_id = orig_doc.get('hash_id')

                people, orgs = extract_entities_from_spacy_doc(spacy_doc, fuser)

                for name in people:
                    if nodes[name]['count'] == 0:
                        person_count += 1
                    nodes[name]['count'] += 1
                    nodes[name]['type'] = 'person'
                    if len(nodes[name]['docs']) < 10:
                        nodes[name]['docs'].append(doc_id)

                for org in orgs:
                    if nodes[org]['count'] == 0:
                        org_count += 1
                    nodes[org]['count'] += 1
                    nodes[org]['type'] = 'org'
                    if len(nodes[org]['docs']) < 10:
                        nodes[org]['docs'].append(doc_id)

                all_entities = list(people) + list(orgs)
                if len(all_entities) >= 2:
                    for name1, name2 in combinations(sorted(all_entities), 2):
                        edges[(name1, name2)]['weight'] += 1
                        if hash_id:
                            edges[(name1, name2)]['doc_ids'].add(hash_id)

                processed += 1

            if processed % progress_interval < batch_size:
                print(f"  Processed {processed}/{total_docs} documents, found {person_count} people, {org_count} orgs...")

            batch_docs = []
            batch_texts = []

    # Process remaining docs
    if batch_docs:
        for spacy_doc, orig_doc in zip(nlp.pipe(batch_texts, batch_size=len(batch_texts)), batch_docs):
            doc_id = str(orig_doc.get('_id'))
            hash_id = orig_doc.get('hash_id')

            people, orgs = extract_entities_from_spacy_doc(spacy_doc, fuser)

            for name in people:
                if nodes[name]['count'] == 0:
                    person_count += 1
                nodes[name]['count'] += 1
                nodes[name]['type'] = 'person'
                if len(nodes[name]['docs']) < 10:
                    nodes[name]['docs'].append(doc_id)

            for org in orgs:
                if nodes[org]['count'] == 0:
                    org_count += 1
                nodes[org]['count'] += 1
                nodes[org]['type'] = 'org'
                if len(nodes[org]['docs']) < 10:
                    nodes[org]['docs'].append(doc_id)

            all_entities = list(people) + list(orgs)
            if len(all_entities) >= 2:
                for name1, name2 in combinations(sorted(all_entities), 2):
                    edges[(name1, name2)]['weight'] += 1
                    if hash_id:
                        edges[(name1, name2)]['doc_ids'].add(hash_id)

            processed += 1

    print(f"Done! Found {person_count} people, {org_count} orgs, {len(edges)} connections.")
    print(f"Entity fuser merged {len(fuser.name_to_canonical)} name variations into {len(fuser.canonical_to_names)} unique entities.")
    return dict(nodes), dict(edges)


def cleanup_merged_names(nodes, min_docs_for_reference=50):
    """
    Remove entities that appear to be two people merged together.
    Uses high-frequency entities as reference to detect merges like "Pruitt Ryan Jackson".
    """
    # Build set of known high-frequency person names
    high_freq_people = {}
    for name, data in nodes.items():
        if data.get('type') == 'person' and data['count'] >= min_docs_for_reference:
            # Store both full name and individual parts
            parts = name.split()
            if len(parts) >= 2:
                high_freq_people[name.lower()] = data['count']
                # Also store last name for matching
                last_name = parts[-1].lower()
                first_name = parts[0].lower()
                if last_name not in high_freq_people or high_freq_people.get(last_name, 0) < data['count']:
                    high_freq_people[last_name] = data['count']
                if first_name not in high_freq_people or high_freq_people.get(first_name, 0) < data['count']:
                    high_freq_people[first_name] = data['count']

    # Also add known surnames
    for surname in ['hupp', 'pruitt', 'jackson', 'clinton', 'beck', 'ford', 'zinke', 'dravis']:
        if surname not in high_freq_people:
            high_freq_people[surname] = 100  # Give them a baseline

    print(f"  Built reference set of {len(high_freq_people)} high-frequency name parts")

    # Common first names that shouldn't appear after a surname
    common_first_names = {
        'sydney', 'ryan', 'scott', 'michael', 'david', 'john', 'james',
        'robert', 'william', 'richard', 'joseph', 'thomas', 'mary', 'jennifer',
        'linda', 'elizabeth', 'barbara', 'susan', 'jessica', 'sarah', 'karen',
        'millan', 'hayley', 'michelle', 'aaron', 'cheryl', 'aurelia', 'nancy',
        'donald', 'barack', 'george', 'hillary', 'joe', 'kamala', 'liz',
        'ray', 'rafael', 'sidney', 'madeline', 'christy', 'mike', 'michele',
    }

    # Words that are clearly not names (appearing after surnames)
    not_names = {
        'invite', 'nice', 'feel', 'group', 'the', 'and', 'or', 'but',
    }

    # Find entities that look like merged names
    to_remove = set()
    for name, data in nodes.items():
        parts = name.split()

        # Check 2-word names for "Lastname Firstname" or "Lastname NotAName" pattern
        if len(parts) == 2:
            first_word = parts[0].lower()
            second_word = parts[1].lower()
            # If first word is a known high-freq surname
            if first_word in high_freq_people and high_freq_people.get(first_word, 0) >= 50:
                # Remove if second word is a common first name (wrong order)
                if second_word in common_first_names:
                    to_remove.add(name)
                    continue
                # Remove if second word is clearly not a name
                if second_word in not_names:
                    to_remove.add(name)
                    continue

        if len(parts) < 3:
            continue  # Need at least 3 words for other checks

        name_lower = name.lower()

        # Check if this looks like "Lastname Firstname Lastname" (two people)
        # e.g., "Pruitt Ryan Jackson" = Pruitt + Ryan Jackson
        for i in range(1, len(parts)):
            potential_second_person = ' '.join(parts[i:]).lower()
            first_part = parts[i-1].lower() if i > 0 else ''

            # Check if the second part matches a known high-freq person
            if potential_second_person in high_freq_people:
                # And the first part is a known surname
                if first_part in high_freq_people:
                    to_remove.add(name)
                    break

            # Also check if last word is a known surname and appears after another name
            if len(parts) >= 3:
                # "Pruitt Ms Hupp" - Pruitt is surname, Hupp is surname
                first_word = parts[0].lower()
                last_word = parts[-1].lower()
                if first_word in high_freq_people and last_word in high_freq_people:
                    # Both first and last words are known names - likely merged
                    if first_word != last_word:  # Not a repeated name
                        to_remove.add(name)
                        break

    print(f"  Identified {len(to_remove)} merged name entities to remove")

    # Remove the merged entities
    for name in to_remove:
        del nodes[name]

    return nodes


def export_to_json(nodes, edges, output_path, min_edge_weight=1, min_node_count=1):
    """Export to JSON for web visualization."""

    # First cleanup merged names
    print("Cleaning up merged names...")
    nodes = cleanup_merged_names(nodes)

    filtered_nodes = {k: v for k, v in nodes.items() if v['count'] >= min_node_count}
    filtered_names = set(filtered_nodes.keys())

    filtered_edges = []
    for k, v in edges.items():
        weight = v['weight']
        doc_ids = list(v.get('doc_ids', set()))
        if weight >= min_edge_weight and k[0] in filtered_names and k[1] in filtered_names:
            filtered_edges.append((k[0], k[1], weight, doc_ids))

    person_count = sum(1 for ndata in filtered_nodes.values() if ndata.get('type', 'person') == 'person')
    org_count = sum(1 for ndata in filtered_nodes.values() if ndata.get('type', 'person') == 'org')

    data = {
        'stats': {
            'people': person_count,
            'orgs': org_count,
            'total': len(filtered_nodes),
            'edges': len(filtered_edges)
        },
        'nodes': [
            {
                'id': name,
                'count': ndata['count'],
                'type': ndata.get('type', 'person'),
            }
            for name, ndata in filtered_nodes.items()
        ],
        'edges': [
            {
                'source': e[0],
                'target': e[1],
                'weight': e[2],
                'doc_ids': e[3]
            }
            for e in filtered_edges
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Saved to {output_path}")
    print(f"  People: {person_count}, Organizations: {org_count}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract network from edgifoia MongoDB using spaCy NER')
    parser.add_argument('--output', '-o', default='edgi_network.json', help='Output filename')
    parser.add_argument('--min-weight', '-w', type=int, default=2, help='Min edge weight')
    parser.add_argument('--min-count', '-c', type=int, default=2, help='Min document count for nodes')

    args = parser.parse_args()

    client = MongoClient('localhost', 27017)
    db = client.edgifoia

    print(f"Min edge weight: {args.min_weight}")
    print(f"Min node count: {args.min_count}")
    print()

    nodes, edges = build_cooccurrence_graph(db)
    export_to_json(nodes, edges, args.output, args.min_weight, args.min_count)

    print("\n=== Summary ===")
    print(f"Total unique entities: {len(nodes)}")
    print(f"Total connections: {len(edges)}")

    top_names = sorted(nodes.items(), key=lambda x: -x[1]['count'])[:20]
    print("\nTop 20 most frequent entities:")
    for name, data in top_names:
        entity_type = data.get('type', 'person')
        print(f"  [{entity_type}] {name}: {data['count']} documents")
