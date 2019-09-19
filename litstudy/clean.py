import yaml
import Levenshtein
import ipywidgets as widgets
from IPython.display import display

# ---------------------------------------
# Cleaning functions
# ---------------------------------------
def get_sources(docset):
    return [d.source for d in docset.docs if d.source is not None]

def get_affiliations(docset, attribute='name'):
    affiliations = []
    for d in docset.docs:
        affiliations += get_affiliations_doc(d, attribute)

    return affiliations

def get_affiliations_doc(doc, attribute='name'):
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

def affiliation_to_type(name):
    name = name.lower()
    pairs = [
        ['universi', 'Academic institute'],
        ['hochschule', 'Academic institute'],
        ['school', 'Academic institute'],
        ['ecole', 'Academic institute'],
        ['institute', 'Academic institute'],
        ['research center', 'Academic institute'],
        ['laboratories', 'Laboratory'],
        ['laboratory', 'Laboratory'],
        ['corporation', 'Corporation'],
        ['corp', 'Corporation'],
        ['ltd', 'Corporation'],
        ['limited', 'Corporation'],
        ['gmbh', 'Corporation'],
        ['ministry', 'Ministry'],
        ['school of', ''],
    ]
    
    for word, affiliation_type in pairs:
        if word in name:
            return affiliation_type
    
    return 'Unknown'

def clean_attributes(plot_callback, docset, x, ax, filename, cleaning_type='affiliations'):
    if filename is not None:
        translation = read_translation_file(filename)

        # Perform translation
        type2replace_f = {'sources': replace_sources, 'affiliations': replace_affiliation_names}
        replace_f = type2replace_f[cleaning_type]
        docset = replace_f(docset, translation)
    else:
        translation = {"translations" :{}, "rejects": []}

    param_passthrough = {   'filename': filename,
                            'cleaning_type': cleaning_type,
                            'docset': docset,
                            'translation': translation,
                            'widgets':
                                {'choice_widget': None, 'text_widget': None, 'custom_widget': None},
                            'plot_params':
                                {'plot_callback': plot_callback, 'x': x, 'ax': ax}
                        }

    start_cleaning(param_passthrough)

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

def start_cleaning(param_passthrough):
    if param_passthrough['cleaning_type'] == 'affiliations':
        get_f = get_affiliations
        remove = ['University', 'Universitat', 'Laboratories', 'Laboratory',
            'National', 'Corporation', 'Technology', 'Science', 'Institute',
            'Ltd.', 'of', 'and']    

    elif param_passthrough['cleaning_type'] == 'sources':
        get_f = get_sources
        remove = ['IEEE', 'th', 'Conference', 'on', 'and', 'Symposium']

    # Get attributes
    attributes = get_f(param_passthrough['docset'])
    
    # Remove duplicates and, for now, remove lists
    attributes = [a for a in attributes if not isinstance(a, list)]
    attributes = list(set(attributes))

    cleaned_attributes = []

    # Remove common words
    for value in attributes:
        for needle in remove:
            value = value.replace(needle, '')
        # Remove excessive whitespace
        value = ' '.join(value.split())
        cleaned_attributes.append(value)

    # Create list of pairs of attributes with similarity score > 0.8
    pairs = []
    for i, at in enumerate(attributes):
        for j, at2 in enumerate(attributes[i+1:]):
            ratio = Levenshtein.ratio(cleaned_attributes[i], cleaned_attributes[i+1+j])
            if ratio > 0.8:
                pairs.append((ratio, at, at2))

    # Sort on similarity score
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

    # Remove scores
    pairs = [(at, at2) for (score, at, at2) in pairs]

    # Remove pairs that are already in the list of rejected merges.
    pairs = filter_rejects(param_passthrough['translation']['rejects'], pairs)

    # Create choice widget or plot
    if len(pairs) > 0:
        options = [pairs[0][0], pairs[0][1], "Don't merge", 'Stop']
        create_widgets(options, None, pairs, param_passthrough)
    else:
        end_cleaning(param_passthrough)

def filter_rejects(rejects, pairs):
    new_pairs = []
    
    for at, at2 in pairs:
        if [at, at2] not in rejects and [at2, at] not in rejects:
            new_pairs.append((at, at2))
    
    return new_pairs

def callback(data, pairs, param_passthrough):
    if isinstance(data, widgets.widgets.widget_string.Text):
        # 'Custom' choisen
        widget = param_passthrough['widgets']['choice_widget']
    else:
        # One of options chosen
        widget = data['owner']
        choice = data['new']

    translation = param_passthrough['translation']
    widget.visible = False
    stop = False

    # Remove this pair from the list
    pairs = pairs[1:]

    if isinstance(data, widgets.widgets.widget_string.Text):
        # 'Custom' choisen
        new = data.value
        widget = param_passthrough['widgets']['choice_widget']

        for old in [widget.options[0], widget.options[1]]:
            translation = add_translation(translation, old, new)
            # Replace pairs with 'old' in list with (new, old) and remove duplicates.
            pairs = [(new, at2) if at == old else (at, at2) for at, at2 in pairs]
            pairs = [(new, at) if at2 == old else (at, at2) for at, at2 in pairs]
            pairs = list(dict.fromkeys(pairs))

    elif choice == widget.options[0]:
        old = widget.options[1]
        new = widget.options[0]
        translation = add_translation(translation, old, new)
        
        # Remove pairs with 'old' from list
        pairs = [(at, at2) for (at, at2) in pairs if (at != old and at2 != old)]

    elif choice == widget.options[1]:
        old = widget.options[0]
        new = widget.options[1]
        translation = add_translation(translation, old, new)

        pairs = [(at, at2) for (at, at2) in pairs if (at != old and at2 != old)]

    elif choice == widget.options[2]:
        old = widget.options[0]
        new = widget.options[1]

        # Add to rejects
        translation['rejects'].append([old, new])
    else:
        stop = True

    param_passthrough['translation'] = translation
    remove_widgets(param_passthrough)

    if len(pairs) <= 0 or stop:
        end_cleaning(param_passthrough)
    else:
        options = [pairs[0][0], pairs[0][1], "Don't merge", 'Stop']
        create_widgets(options, data, pairs, param_passthrough)

def add_translation(translation, old, new):
    translation['translations'][old] = new

    # If 'new' was translated to something else previously, immediately
    # translate 'old' to that too now.
    try:
        new = translation['translations'][new]
        translation['translations'][old] = new
    except KeyError:
        pass

    # If something was translated to 'old' previously, now translate it to 'new'.
    for key in translation['translations'].keys():
        if translation['translations'][key] == old:
            translation['translations'][key] = new

    return translation

def create_widgets(options, data, pairs, param_passthrough):
    param_passthrough['widgets']['text_widget'] = create_text_widget(param_passthrough)
    param_passthrough['widgets']['choice_widget'] = create_choice_widget(options, data, pairs, param_passthrough)
    param_passthrough['widgets']['custom_widget'] = create_custom_widget(data, pairs, param_passthrough)

def create_text_widget(param_passthrough):
    text = """If the following two """ + param_passthrough['cleaning_type'] + \
            """ are the same, please pick which way of writing you prefer.
            <br>If the they are distinct, pick \"Don't merge\".
            <br>If you wish to stop the cleaning process, click "Stop\".
            <br>To use a new name for both options,
            enter a new name in the text box and press 'enter'."""

    widget = widgets.HTML(value=text)
    display(widget)
    return widget

def create_choice_widget(options, data, pairs, param_passthrough):
    widget = widgets.ToggleButtons(options=options, disabled=False, value=None)
    widget.style.button_width='100%'
    widget.observe(lambda data: callback(data, pairs, param_passthrough), 'value')
    display(widget)
    return widget

def create_custom_widget(data, pairs, param_passthrough):
    widget = widgets.Text(description='Custom:', disabled=False)
    widget.on_submit(lambda data: callback(data, pairs, param_passthrough))
    display(widget)
    return widget

def remove_widgets(param_passthrough):
    remove_widget(param_passthrough['widgets']['text_widget'])
    remove_widget(param_passthrough['widgets']['choice_widget'])
    remove_widget(param_passthrough['widgets']['custom_widget'])
    param_passthrough['widgets']['text_widget'] = None
    param_passthrough['widgets']['choice_widget'] = None
    param_passthrough['widgets']['custom_widget'] = None

def remove_widget(widget):
    widget.close()
    del widget

def end_cleaning(param_passthrough):
    filename = param_passthrough['filename']
    docset = param_passthrough['docset']
    translation = param_passthrough['translation']

    # Perform translation
    type2replace_f = {'sources': replace_sources, 'affiliations': replace_affiliation_names}
    replace_f = type2replace_f[param_passthrough['cleaning_type']]
    param_passthrough['docset'] = replace_f(docset, translation)

    write_translation_file(filename, translation)
    
    plot_params = param_passthrough['plot_params']
    plot_callback = plot_params['plot_callback']
    plot_callback(param_passthrough['docset'], param_passthrough['plot_params']['x'],
        param_passthrough['plot_params']['ax'], clean=False)

def replace_sources(docset, translation):
    for doc in docset:
        try:
            new = translation['translations'][doc.source]
            doc.source = new
        except (KeyError, TypeError) as e:
            # Source is None, list or no translation is found.
            pass
    return docset

def replace_affiliation_names(docset, translation):
    for doc in docset:
        for author in doc.authors:
            if author.affiliations is not None:
                for affiliation in author.affiliations:
                    try:
                        new = translation['translations'][affiliation.name]
                        affiliation.name = new
                    except (KeyError, TypeError, AttributeError) as e:
                        # Affiliation or name is None or a list or no
                        # translation is found.
                        pass
    return docset
