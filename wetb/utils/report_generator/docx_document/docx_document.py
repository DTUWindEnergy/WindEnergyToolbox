#!/usr/bin/env python

"""
This file makes a .docx (Word 2007) file from scratch, showing off most of the
features of python-docx.

If you need to make documents from scratch, you can use this file as a basis
for your work.

Part of Python's docx module - http://github.com/mikemaccana/python-docx
See LICENSE for licensing information.
"""

import os
import re
import shutil

from mmpe.docx_document.docx import picture, table, caption, pagebreak
from mmpe.docx_document import docx
from lxml import etree

from mmpe.functions.process_exec import pexec
from contextlib import contextmanager
from mmpe.functions.deep_coding import to_str
from mmpe.io.make_dirs import make_dirs


re_bullet_lst = re.compile("[+,\-,*] (.*)")
re_number_lst = re.compile("[\d]+\. (.*)")
inkscape_path = os.path.relpath(os.path.join(os.path.dirname(__file__.replace("library.zip", '')), "inkscape/inkscape.exe"))
pp = lambda t : etree.tostring(t, pretty_print=True)

def vector_graphics_support():
    return os.path.isfile(inkscape_path)

def prefered_graphic_format(vector=False):
    if vector and vector_graphics_support():
        return "pdf"
    else:
        return "png"

class ConversionError(Exception):
    pass


def DocxDocument(doc, title='', subject='', creator='', keywords=[], page_margins={'top':2, 'bottom':2, 'left':3, 'right':2}, delete_closed=False):
    if isinstance(doc, str):
        doc = DocxDocumentClass(doc, title, subject, creator, keywords, page_margins, delete_closed)
        doc.open_on_exit_lst.append(True)
    else:
        doc.open_on_exit_lst.append(False)
    return doc

class DocxDocumentClass(object):
    open_on_exit_lst = [False]
    def __init__(self, filename, title='', subject='', creator='', keywords=[], page_margins={'top':2, 'bottom':2, 'left':3, 'right':2}, delete_closed=False):
        if "%" in filename:
            self.filename = None
            for i in range(100):
                f = filename % i
                template_dir = os.path.join(os.path.dirname(f), "docx_template_" + os.path.splitext(os.path.basename(f))[0])
                if os.path.isfile(f) or os.path.isdir(template_dir):
                    if delete_closed:
                        try:
                            os.remove(f)
                            if self.filename is None:
                                self.filename = f
                        except:
                            pass
                else:
                    if self.filename is None:
                        self.filename = f
        else:
            self.filename = filename
        try:
            make_dirs(self.filename)
            with open(self.filename, 'w'):
                pass
        except PermissionError as e:
            raise Warning(str(e))

        self.template_dir = os.path.join(os.path.dirname(self.filename), "docx_template_" + os.path.splitext(os.path.basename(self.filename))[0])
        if os.path.isdir(self.template_dir):
            shutil.rmtree(self.template_dir)
        shutil.copytree(os.path.join(os.path.dirname(__file__.replace("library.zip", '')), 'docx-template_clean'), self.template_dir)
        docx.template_dir = self.template_dir

        self.title = title
        self.subject = subject
        self.creator = creator
        self.keywords = keywords

        # Default set of relationshipships - the minimum components of a document
        self.relationships = docx.relationshiplist()
        self.imagefiledict = {}


        # Make a new document tree - this is the main part of a Word document
        self.document = docx.newdocument(page_margins=page_margins)

        # This xpath location is where most interesting content lives
        self.body = self.document.xpath('/w:document/w:body', namespaces=docx.nsprefixes)[0]
        self.h = self.h1 = self.append_heading
        self.h2 = lambda s : self.append_heading(s, 2)
        self.h3 = lambda s : self.append_heading(s, 3)
        self.p = self.append_paragraph
        self.n = lambda s : self.append_paragraph([(s, 'ns')])
        self.i = lambda s : self.append_paragraph([(s, 'i')])
        self.b = lambda s : self.append_paragraph([(s, 'b')])
        self.new_page = lambda : self.body.append(pagebreak())
        self.table = docx.table
        self.paragraph = docx.paragraph
        self.caption = docx.caption
        self.heading = docx.heading

    def write_access(self):
        try:
            with open(self.filename, 'a+'):
                pass
            return True
        except IOError:
            return False


    def close(self):
        if os.path.isdir(self.template_dir):
            shutil.rmtree(self.template_dir)



    def search(self, string):
        return docx.search(self.body, string)

#    def replace(self, find_string, replace_string):
#        replace(self.body, find_string, replace_string)


    def append_heading(self, string, level=1):
        self.body.append(docx.heading(string, level))

    def append_paragraph(self, string, style='BodyText', breakbefore=False, jc='left', spacing={'before':0, 'after':6}, font_size=12):
        self.body.append(docx.paragraph(string, style, breakbefore, jc, spacing, font_size))

    def append_list(self, item_list, style):
        for item in item_list:
            self.append_paragraph(item, style)

    def append_numberlist(self, item_list):
        self.append_list(item_list, 'ListNumber')

    def append_bulletlist(self, item_list):
        self.append_list(item_list, 'ListBullet')

    def append_caption(self, tag, caption_text):
        self.body.append(caption("Figure", caption_text))

    def picture(self, path, title="", pixelwidth=None, pixelheight=None):
        global inkscape_path
        _, ext = os.path.splitext(path.lower())
        if ext in (".svg", '.svgz', '.eps', '.pdf'):
            emf_path = path.replace(ext, '.emf')
            args = [inkscape_path, path, '--export-emf=%s' % emf_path]
            returncode, stdout, stderr, cmd = pexec(args)
            if returncode != 0 or not os.path.exists(emf_path):
                inkscape_path = ""
                raise ConversionError("%s\n%s" % (stdout, stderr))
            #succeded
            path = emf_path

        self.relationships, picpara = picture(self.relationships, path, title, pixelwidth, pixelheight)
        return picpara


    def append_picture(self, path, caption_text="", pixelwidth=None, pixelheight=None):
        self.body.append(self.picture(path, caption_text, pixelwidth, pixelheight))
#        if caption_text != "":
#            self.body.append(caption("Figure", caption_text))



    def append_table(self, lst_of_lst, header=None, first_column=None, corner="", heading=False, colw=None, cwunit='dxa',
                     tblw=0, twunit='auto',
                     tblmargin={'left':.19, 'right':.19},
                     borders={"all":{"color":'auto', 'val':'single', 'sz':'4'}},
                     column_style={'all':{'font_size':10, 'spacing':{'before':0, 'after':0}}}):
        """
        @param list contents: A list of lists describing contents. Every item in
                      the list can be a string or a valid XML element
                      itself. It can also be a list. In that case all the
                      listed elements will be merged into the cell.
        @param bool heading:  Tells whether first line should be treated as
                              heading or not
        @param list colw:     list of integer column widths specified in wunitS.
        @param str  cwunit:   Unit used for column width:
                                'pct'  : fiftieths of a percent
                                'dxa'  : twentieths of a point
                                'nil'  : no width
                                'auto' : automagically determined
        @param int  tblw:     Table width
        @param str  twunit:   Unit used for table width. Same possible values as
                              cwunit.
        @param dict borders:  Dictionary defining table border. Supported keys
                              are: 'top', 'left', 'bottom', 'right',
                              'insideH', 'insideV', 'all'.
                              When specified, the 'all' key has precedence over
                              others. Each key must define a dict of border
                              attributes:
                                color : The color of the border, in hex or
                                        'auto'
                                space : The space, measured in points
                                sz    : The size of the border, in eighths of
                                        a point
                                val   : The style of the border, see
                    http://www.schemacentral.com/sc/ooxml/t-w_ST_Border.htm
        @param list celstyle: Specify the style for each colum, list of dicts.
                              supported keys:
                              'align' : specify the alignment, see paragraph
                                        documentation."""
        m = len(lst_of_lst)
        n = len(lst_of_lst[0])
        if header:

            if first_column:
                lst_of_lst = [[corner] + header] + [[fc] + lst for fc, lst in zip(first_column, lst_of_lst)]
            else:
                lst_of_lst = [header] + [ lst for lst in lst_of_lst]
        lst_of_lst = [[(str(v), v)[isinstance(v, (etree._Element, list, tuple))] for v in lst] for lst in lst_of_lst]

        self.body.append(table(lst_of_lst, heading, colw, cwunit, tblw, tblmargin, twunit, borders, column_style))



    def append_markdown(self, s):
        number_lst = []
        bullet_lst = []

        for l in s.split("\n"):
            #bullet list
            if re_bullet_lst.match(l) is not None:
                bullet_lst.append(re_bullet_lst.match(l).groups()[0])
                continue
            elif len(bullet_lst) > 0:
                self.append_bulletlist(bullet_lst)
                bullet_lst = []
            # number list
            if re_number_lst.match(l) is not None:
                number_lst.append(re_number_lst.match(l).groups()[0])
                continue
            elif len(number_lst) > 0:
                self.append_numberlist(number_lst)
                number_lst = []



            #headings
            if l.startswith("#"):
                for i in [3, 2, 1]:
                    if l.startswith("#"*i):
                        self.append_heading(l[i:], i)
                        break

            elif (' *' in l and "* " in l) or (" _" in l and "_ " in l):
                def styleit(text_lst, style_lst, tag, style):
                    t_lst = []
                    s_lst = []
                    for t, s in (zip(text_lst, style_lst)):
                        while t.count(tag) > 2:  #" " + tag in t and tag + " " in t:
                            before, rest = t.split(tag, 1)
                            style_text, t = rest.split(tag, 1)
                            t_lst.append(before);s_lst.append(s)
                            t_lst.append(style_text);s_lst.append(style)
                        else:
                            t_lst.append(t);s_lst.append(s)
                    return t_lst, s_lst
                text_lst, style_lst = [l], ['n']
                text_lst, style_lst = styleit(text_lst, style_lst, "**", 'b')
                text_lst, style_lst = styleit(text_lst, style_lst, "__", 'b')
                text_lst, style_lst = styleit(text_lst, style_lst, "*", 'i')
                text_lst, style_lst = styleit(text_lst, style_lst, "_", 'i')
                self.append_paragraph(list(zip(text_lst, style_lst)))
            else:
                self.n(l)

    def landscape(self, page_margins={'top':2, 'bottom':2, 'left':3, 'right':2}):
        self.body.append(pagebreak('section', page_margins, portrait=False))

    def portrait(self, page_margins={'top':2, 'bottom':2, 'left':3, 'right':2}):
        self.body.append(pagebreak('section', page_margins, portrait=True))

    def save(self, filename=None):
        if filename is not None:
            self.filename = filename

        coreprops = docx.coreproperties(title=self.title,
                                   subject=self.subject,
                                   creator=self.creator,
                                   keywords=self.keywords)
        appprops = docx.appproperties()
        contenttypes = docx.contenttypes()
        websettings = docx.websettings()
        wordrelationships = docx.wordrelationships(self.relationships)
        # Save our document
        docx.savedocx(self.document, coreprops, appprops, contenttypes, websettings,
                 wordrelationships, self.filename)

    def open(self):
        self.save()
        self.close()
        import subprocess
        subprocess.Popen(os.path.realpath(self.filename), shell=True, cwd=os.getcwd())

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.open_on_exit_lst.pop():
            self.open()



if __name__ == '__main__':

#    doc1 = DocxDocument('test%d.docx')
#    doc1.h1("first")
#    doc1.open()
    with DocxDocument('test%d.docx', delete_closed=True) as doc:
        doc.p("hej")

#    doc3 = DocxDocument('test%d.docx')
#    doc3.h1("thirda")
#    with DocxDocument(doc3) as doc:
#        doc.h1("thirdb")

#    doc = DocxDocument("test%d.docx")
#    doc.append_table([['1', '2'], ['3', '4']], header=['a', 'b'], first_column=['c', 'd'], corner='table')
#    doc.p("")
#    doc.append_table([['1', '2'], ['3', '4']], header=['a', 'b'], first_column=['c', 'd'])
#    doc.p("")
#    doc.append_table([['1', '2'], ['3', '4']], header=['a', 'b'])
#    doc.p("")
#    doc.append_table([['1', '2'], ['3', '4']], first_column=['a', 'b'])

    #table_xml = doc.table([['A1'], ['B1']], heading=False, column_style={'all':{'font_size':20, 'spacing':{'before':0, 'after':0}}})
#    p = doc.paragraph("Test", font_size=30)
#
#    doc.body.append(p)
#
#    doc.n("")
#
#    # Append two headings and a paragraph
#    doc.h("Welcome to Python's docx module")
#    doc.h2('Make and edit docx in 200 lines of pure Python')
#    doc.n('The module was created when I was looking for a '
#        'Python support for MS Word .doc files on PyPI and Stackoverflow. '
#        'Unfortunately, the only solutions I could find used:')
#
#    doc.i('For those of us who prefer something simpler')
#    doc.b('I made docx.')
#
#
#    # Add a numbered list
#    points = [ 'COM automation'
#             , '.net or Java'
#             , 'Automating OpenOffice or MS Office'
#             ]
#    doc.append_numberlist(points)
#
#    doc.append_picture('image1.png', "This is a picture")
#
#    doc.append_table([ [doc.picture('image1.png'), 'A2', 'A3'],
#                       ['B1', 'B2', 'B3'],
#                       ['C1', 'C2', 'C3']], False)
#
##   doc.append_picture('../myplot.pdf', "test", 100, 100)
#    doc.append_markdown("""#Welcome to Python's docx module
###Make and edit docx in 200 lines of pure Python
#The module was created when I was looking for a Python support for MS Word .doc files on PyPI and Stackoverflow. Unfortunately, the only solutions I could find used:
#*For those of us who prefer something simpler* **I made docx.**
#
#4. First
#5. Second
#6. Third
#
#- first
#* second
#+ third
#""")


    #doc.append_heading('Making documents', 2)
#    body.append(paragraph('The docx module has the following features:'))
#
#    # Add some bullets
#    points = ['Paragraphs', 'Bullets', 'Numbered lists',
#              'Multiple levels of headings', 'Tables', 'Document Properties']
#    for point in points:
#        body.append(paragraph(point, style='ListBullet'))
#
#    body.append(paragraph('Tables are just lists of lists, like this:'))
#    # Append a table
#    tbl_rows = [ ['A1', 'A2', 'A3']
#               , ['B1', 'B2', 'B3']
#               , ['C1', 'C2', 'C3']
#               ]
#    body.append(table(tbl_rows))
#
#    body.append(heading('Editing documents', 2))
#    body.append(paragraph('Thanks to the awesomeness of the lxml module, '
#                          'we can:'))
#    points = [ 'Search and replace'
#             , 'Extract plain text of document'
#             , 'Add and delete items anywhere within the document'
#             ]
#    for point in points:
#        body.append(paragraph(point, style='ListBullet'))
#
#    # Add an image
#    relationships, picpara = picture(relationships, 'image1.png',
#                                     'This is a test description')
#    body.append(picpara)
#
#    # Search and replace
#    print 'Searching for something in a paragraph ...',
#    if search(body, 'the awesomeness'):
#        print 'found it!'
#    else:
#        print 'nope.'
#
#    print 'Searching for something in a heading ...',
#    if search(body, '200 lines'):
#        print 'found it!'
#    else:
#        print 'nope.'
#
#    print 'Replacing ...',
#    body = replace(body, 'the awesomeness', 'the goshdarned awesomeness')
#    print 'done.'
#
#    # Add a pagebreak
#    body.append(pagebreak(type='page', orient='portrait'))
#
#    body.append(heading('Ideas? Questions? Want to contribute?', 2))
#    body.append(paragraph('Email <python.docx@librelist.com>'))

#    # Create our properties, contenttypes, and other support files
#    title = 'Python docx demo'
#    subject = 'A practical example of making docx from Python'
#    creator = 'Mike MacCana'
#    keywords = ['python', 'Office Open XML', 'Word']

#    coreprops = coreproperties(title=title, subject=subject, creator=creator,
#                               keywords=keywords)
#    appprops = appproperties()
#    contenttypes = contenttypes()
#    websettings = websettings()
#    wordrelationships = wordrelationships(relationships)
#
#    # Save our document
#    savedocx(document, coreprops, appprops, contenttypes, websettings,
#             wordrelationships, 'Welcome to the Python docx module.docx')
#    doc.save()
#    doc.close()
#    doc.open()
#
