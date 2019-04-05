# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from collections import OrderedDict
import pytest
from copy import deepcopy
from numpy import array
import astropy.units as u
from astropy.table import QTable
from ..core import DataClass, conf


def data_path(filename):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    return os.path.join(data_dir, filename)


def test_get_set():
    """ test the get and set methods"""

    data = DataClass.from_dict(
        [OrderedDict((('a', 1), ('b', 4), ('c', 'a'))),
         OrderedDict((('a', 2), ('b', 5), ('c', 'b'))),
         OrderedDict((('a', 3), ('b', 6), ('c', 'c')))])

    # get a single column
    assert len(data['a']) == 3

    # mask rows
    masked = data[[True, False, False]]
    assert len(masked) == 1
    assert masked['b'][0] == 4

    # get list of rows
    shortened = data[[0, 1]]
    assert len(shortened) == 2

    # modify an existing column
    data['a'][:] = [0, 0, 0]
    assert data['a'][0] == 0

    with pytest.raises(AttributeError):
        data.d

    with pytest.raises(KeyError):
        data['d']


def test_creation_single():
    """ test the creation of DataClass objects from dicts or arrays;
    single row only"""

    ground_truth = QTable([[1], [2], ['test']], names=('a', 'b', 'c'))

    test_init = DataClass(OrderedDict([('a', 1), ('b', 2), ('c', 'test')]))
    assert test_init.table == ground_truth

    test_dict = DataClass.from_dict(
        OrderedDict([('a', 1), ('b', 2), ('c', 'test')]))
    assert test_dict.table == ground_truth

    test_array = DataClass.from_array([[1], [2], ['test']],
                                      names=('a', 'b', 'c'))
    assert test_array.table == ground_truth

    # test creation fails
    with pytest.raises(TypeError):
        DataClass.from_dict(True)

    with pytest.raises(TypeError):
        DataClass.from_array(True)


def test_creation_multi():
    """ test the creation of DataClass objects from dicts or arrays;
    multiple rows"""

    ground_truth = QTable([[1, 2, 3], [4, 5, 6], ['a', 'b', 'c']],
                          names=('a', 'b', 'c'))

    test_dict = DataClass.from_dict(
        [OrderedDict((('a', 1), ('b', 4), ('c', 'a'))),
         OrderedDict((('a', 2), ('b', 5), ('c', 'b)'))),
         OrderedDict((('a', 3), ('b', 6), ('c', 'c')))])
    assert all(test_dict.table == ground_truth)

    test_array = DataClass.from_array([[1, 2, 3], [4, 5, 6],
                                       ['a', 'b', 'c']],
                                      names=('a', 'b', 'c'))
    assert all(test_array.table == ground_truth)

    test_table = DataClass.from_table(ground_truth)
    assert all(test_table.table == ground_truth)

    # test reading data from file
    ra = [10.223423, 10.233453, 10.243452]
    dec = [-12.42123, -12.41562, -12.40435]
    epoch = 2451523.5 + array([0.1234, 0.2345, 0.3525])
    file_ground_truth = DataClass.from_array(
        [ra, dec, epoch], names=['ra', 'dec', 't'])

    test_file = DataClass.from_file(data_path('test.txt'),
                                    format='ascii')

    assert all(file_ground_truth.table == test_file.table)

    # test failing if columns have different lengths
    with pytest.raises(ValueError):
        test_dict = DataClass.from_dict(
            OrderedDict([(('a', 1), ('b', 4), ('c', 'a')),
                         (('a', 2), ('b', 5), ('c', 'b)')),
                         (('a', 3), ('b', 6), ('c', 7))]))

    with pytest.raises(ValueError):
        test_array = DataClass.from_array([[1, 2, 3], [4, 5, 6], ['a', 'b']],
                                          names=('a', 'b', 'c'))


def test_units():
    """ test units on multi-row tables """

    ground_truth = QTable([[1, 2, 3]*u.Unit('m'),
                           [4, 5, 6]*u.m/u.s,
                           ['a', 'b', 'c']],
                          names=('a', 'b', 'c'))

    assert ((ground_truth['a']**2).unit == 'm2')

    test_dict = DataClass.from_dict(
        [OrderedDict((('a', 1*u.m), ('b', 4*u.m/u.s), ('c', 'a'))),
         OrderedDict((('a', 2*u.m), ('b', 5*u.m/u.s), ('c', 'b'))),
         OrderedDict((('a', 3*u.m), ('b', 6*u.m/u.s), ('c', 'c')))])
    assert all(test_dict.table == ground_truth)

    test_array = DataClass.from_array([[1, 2, 3]*u.m,
                                       [4, 5, 6]*u.m/u.s,
                                       ['a', 'b', 'c']],
                                      names=('a', 'b', 'c'))
    assert all(test_array.table == ground_truth)


def test_add():
    """ test adding rows and columns to an existing table """

    tab = DataClass.from_dict(
        [OrderedDict((('a', 1*u.m), ('b', 4*u.m/u.s), ('c', 'a'))),
         OrderedDict((('a', 2*u.m), ('b', 5*u.m/u.s), ('c', 'b'))),
         OrderedDict((('a', 3*u.m), ('b', 6*u.m/u.s), ('c', 'c')))])

    # adding single rows

    tab.add_rows([4*u.m, 7*u.m/u.s, 'd'])

    with pytest.raises(ValueError):
        tab.add_rows([4*u.s, 7*u.m/u.s, 'd'])  # fails: wrong unit

    tab.add_rows({'a': 5*u.m, 'b': 8*u.m/u.s, 'c': 'e'})

    with pytest.raises(ValueError):
        tab.add_rows({'a': 5*u.m, 'b': 8*u.m/u.s})  # fails: incomplete

    # ignore superfluent columns
    tab.add_rows({'a': 6*u.m, 'b': 9*u.m/u.s, 'c': 'f', 'd': 'not existent'})

    # adding multiple rows

    tab.add_rows(([7*u.m, 10*u.m/u.s, 'g'],
                  [8*u.m, 11*u.m/u.s, 'h']))

    tab.add_rows([{'a': 9*u.m, 'b': 12*u.m/u.s, 'c': 'i'},
                  {'a': 10*u.m, 'b': 13*u.m/u.s, 'c': 'j'}])

    assert all(tab['a']**2 == [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]*u.m*u.m)

    # adding columns

    tab.add_column(array([10, 20, 30, 40, 50,
                          60, 70, 80, 90, 100])*u.kg/u.um, name='d')

    print(tab.table)

    assert tab['d'][0] == 10*u.kg/u.um

    with pytest.raises(ValueError):
        tab.add_column(array([10, 20, 30, 40, 50,
                              60, 70, 80, 90])*u.kg/u.um, name='e')

    assert tab.add_rows(tab) == 20


def test_alternative_name_uniqueness():
    """test the uniqueness of alternative field names"""
    from ..core import conf

    assert (len(sum(conf.fieldnames, [])) ==
            len(set(sum(conf.fieldnames, []))))

    storage = (deepcopy(conf.fieldnames), deepcopy(conf.fieldname_idx))

    with pytest.raises(AssertionError):
        # repeat existing fieldname should raise Error
        conf.fieldnames.append(['i'])
        assert (len(sum(conf.fieldnames, [])) ==
                len(set(sum(conf.fieldnames, []))))

    # revert changes to conf.fieldnames
    conf.fieldnames = storage[0]
    conf.fieldname_idx = storage[1]


def test_translate_columns():
    """test function that translates column names"""

    storage = (deepcopy(conf.fieldnames), deepcopy(conf.fieldname_idx))
    conf.fieldnames = [['z', 'a']]
    conf.fieldname_idx = {}
    for idx, field in enumerate(conf.fieldnames):
        for alt in field:
            conf.fieldname_idx[alt] = idx

    tab = DataClass.from_dict(
        [OrderedDict((('a', 1*u.m), ('b', 4*u.m/u.s), ('c', 'a'))),
         OrderedDict((('a', 2*u.m), ('b', 5*u.m/u.s), ('c', 'b'))),
         OrderedDict((('a', 3*u.m), ('b', 6*u.m/u.s), ('c', 'c')))])

    assert tab._translate_columns(['a', 'b', 'c']) == ['a', 'b', 'c']
    assert tab._translate_columns(['z', 'b', 'c']) == ['a', 'b', 'c']

    with pytest.raises(KeyError):
        tab._translate_columns(['x'])

    # revert changes to conf.fieldnames
    conf.fieldnames = storage[0]
    conf.fieldname_idx = storage[1]


def test_indexing():
    """make sure that indexing functionality is not compromised through
    column name translation"""

    tab = DataClass.from_dict(
        [OrderedDict((('a', 1*u.m), ('b', 4*u.m/u.s), ('c', 'a'))),
         OrderedDict((('a', 2*u.m), ('b', 5*u.m/u.s), ('c', 'b'))),
         OrderedDict((('a', 3*u.m), ('b', 6*u.m/u.s), ('c', 'c')))])

    assert list(tab['a'].data) == [1, 2, 3]
    assert list(tab['a', 'b']['a'].data) == [1, 2, 3]
    assert len(tab[tab['a'] < 3*u.m]) == 2


def test_field_conversion():
    """test field conversion functions"""

    tab = DataClass.from_dict(OrderedDict((('d', 1*u.m),
                                           ('b', 4*u.m/u.s),
                                           ('c', 'a'))))

    assert tab['d'] == 1*u.m
    assert tab['diameter'] == 1*u.m
    assert tab['R'] == 0.5*u.m
    assert tab['radius'] == 0.5*u.m

    tab = DataClass.from_dict(OrderedDict((('R', 1*u.m),
                                           ('b', 4*u.m/u.s),
                                           ('c', 'a'))))

    assert tab['d'] == 2*u.m
    assert tab['R'] == 1*u.m

    # test something that is not defined anywhere in data.conf
    with pytest.raises(KeyError):
        tab = DataClass.from_dict(OrderedDict((('a', 1*u.m),
                                               ('b', 4*u.m/u.s),
                                               ('c', 'a'))))
        tab['bullpoop']

    # test something that is defined anywhere in data.conf
    with pytest.raises(KeyError):
        tab = DataClass.from_dict(OrderedDict((('a', 1*u.m),
                                               ('b', 4*u.m/u.s),
                                               ('c', 'a'))))
        tab['radius']


def test_expand():
    """tests for sbpy.data.DataClass.expand"""

    tab = DataClass.from_dict(OrderedDict((('a', [1, 2, 3]*u.m),
                                           ('b', [4, 5, 6]*u.m/u.s),
                                           ('c', [7, 8, 9]*u.kg))))

    # apply nested sequence
    tab.expand([[1], [2, 3], [4, 5, 6]], 'test', unit='Hz')

    assert all(tab['test'] == [1., 2., 3., 4., 5., 6.]*u.Hz)

    # apply flat sequence, no unit
    tab.expand(['a', 'b', 'c', 'd', 'e', 'f'], 'test2')

    assert all(tab['test2'] == ['a', 'b', 'c', 'd', 'e', 'f'])

    assert tab['a'][0] == 1.*u.m
    assert tab['b'][1] == 5.*u.m/u.s
    assert tab['c'][2] == 8*u.kg
    assert tab['test'][3] == 4.*u.Hz
    assert tab['test2'][4] == 'e'

    # wrong number of elements (not matching len(self.table))
    with pytest.raises(RuntimeError):
        tab.expand([[1], [2, 3], [4, 5, 6]], 'test2', unit='Hz')

    # add data from Quantity; check unit
    tab.expand([1, 2, 3, 4, 5, 6]*u.s, 'test3')
    assert tab['test3'][0].unit == u.s
