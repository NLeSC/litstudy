import yaml
import Levenshtein
import ipywidgets as widgets
from IPython.display import display

# ---------------------------------------
# Affiliation cleaning functions
# ---------------------------------------

def get_affiliations(doc, attribute='name'):
    # Put affiliations of all authors in one list.
    affiliation_lists = [a.affiliations for a in doc.authors]

    # Remove 'None' affialiations
    affiliations = [x for x in affiliation_lists if x is not None]

    # Flatten lists
    affiliations = [y for x in affiliations for y in x]

    if attribute == 'country':
        # Get affiliation countries and remove 'None' values
        affiliations = [af.country for af in affiliations if af.country is not None]
    elif attribute == 'affiliation_type':
        # Get affiliation names
        affiliations = [af.name for af in affiliations]
        affiliations = [affiliation_to_type(x) for x in affiliations]
    else:
        # Get affiliation names
        affiliations = [af.name for af in affiliations]

    # Remove duplicates (2 authors with same affiliation/country
    # results in 1 count for that affiliation/country).
    return set(affiliations)

def read_translation_file(filename):
    translation = {"translations" :{}, "rejects": []}

    try:
        with open(filename, 'r') as f:
            # yaml.dump(data, outfile)#default_flow_style=False)
            translation = yaml.safe_load(f)
    except FileNotFoundError:
        pass    

    if translation is None:
        translation = {"translations" :{}, "rejects": []}

    return translation

def write_translation_file(filename, translation):
    with open(filename, 'w') as f:
        yaml.dump(translation, f)#default_flow_style=False)

def filter_rejects(rejects, pairs):
    new_pairs = []
    
    for score, af, af2 in pairs:
        if [af, af2] not in rejects and [af2, af] not in rejects:
            new_pairs.append((score, af, af2))
    
    return new_pairs

def clean_affiliations(plot_callback, docset, x, ax, filename):
    if filename is not None:
        translation = read_translation_file(filename)
        docset = replace_affiliation_names(docset, translation)
        start_clean_affiliations(docset, translation, filename, {'plot_callback': plot_callback, 'x': x, 'ax': ax})
    else:
        start_clean_affiliations(docset, {"translations" :{}, "rejects": []}, filename, {'plot_callback': plot_callback, 'x': x, 'ax': ax})

def start_clean_affiliations(docset, translation, filename, plot_params):
    # Get affiliation names
    affiliation_names = []
    for d in docset.docs:
        affiliation_names += get_affiliations(d)
    affiliation_names = list(set(affiliation_names))

    remove = ['University', 'Universitat', 'Laboratories', 'Laboratory',
        'National', 'Corporation', 'Technology', 'Science', 'Institute',
        'Ltd.', 'of', 'and']

    new_names = []

    # Remove common words
    for name in affiliation_names:
        for needle in remove:
            name = name.replace(needle, '')
        # Remove excessive whitespace
        name = ' '.join(name.split())
        new_names.append(name)

    # Create list of pairs of affiliation names with similarity score > 0.9
    pairs = []
    for i, af in enumerate(affiliation_names):
        for j, af2 in enumerate(affiliation_names[i+1:]):
            ratio = Levenshtein.ratio(new_names[i], new_names[i+1+j])
            if ratio > 0.8:
                pairs.append((ratio, af, af2))

    # Sort on similarity score
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

    # Remove pairs that are already in the list of rejected merges.
    pairs = filter_rejects(translation['rejects'], pairs)

    # Create choice widget or plot
    if len(pairs) > 0:
        options = [pairs[0][1], pairs[0][2], "Don't merge", 'Stop']
        create_widget(options, None, pairs, translation, filename, docset, plot_params)
    else:
        end_affiliation_cleaning(docset, translation, filename, plot_params)

def end_affiliation_cleaning(docset, translation, filename, plot_params):
    docset = replace_affiliation_names(docset, translation)
    write_translation_file(filename, translation)
    plot_callback = plot_params['plot_callback']
    plot_callback(docset, plot_params['x'], plot_params['ax'], clean=False)

def create_widget(options, data, pairs, translation, filename, docset, plot_params):
    widget = widgets.ToggleButtons(
        options=options,
        description='Speed:',
        disabled=False,
        value=None
    )
    widget.style.button_width='100%'
    display(widget)
    widget.observe(lambda data: callback(data, pairs, translation, filename, docset, plot_params), 'value')
    return widget

def remove_widget(widget):
    widget.close()
    del widget

def replace_affiliation_names(docset, translation):
    for old in translation['translations'].keys():
        new = translation['translations'][old]
        for doc in docset:
            for author in doc.authors:
                if author.affiliations is not None:
                    for affiliation in author.affiliations:
                        if affiliation is not None and affiliation.name == old:
                            affiliation.name = new
    return docset

def add_translation(translation, old, new):
    translation['translations'][old] = new

    for key in translation['translations'].keys():
        if translation['translations'][key] == old:
            translation['translations'][key] = new

    return translation

def callback(data, pairs, translation, filename, docset, plot_params):
    widget = data['owner']
    choice = data['new']
    widget.visible = False
    stop = False

    if choice == widget.options[0]:
        old = widget.options[1]
        new = widget.options[0]
        translation = add_translation(translation, old, new)
        
        # Remove this pairs and pairs with 'old' from list
        pairs = pairs[1:]
        pairs = [(ratio, af, af2) for (ratio, af, af2) in pairs if (af != old and af2 != old)]
    elif choice == widget.options[1]:
        old = widget.options[0]
        new = widget.options[1]
        translation = add_translation(translation, old, new)

        pairs = pairs[1:]
        pairs = [(ratio, af, af2) for (ratio, af, af2) in pairs if (af != old and af2 != old)]
    elif choice == widget.options[2]:
        old = widget.options[0]
        new = widget.options[1]

        # Add to rejects
        translation['rejects'].append([old, new])
        pairs = pairs[1:]
    else:
        stop = True

    if len(pairs) <= 0:
        stop = True

    remove_widget(widget)

    if stop:
        end_affiliation_cleaning(docset, translation, filename, plot_params)
    else:
        new_options = [pairs[0][1], pairs[0][2], "Don't merge", 'Stop']
        create_widget(new_options, data, pairs, translation, filename, docset, plot_params)

