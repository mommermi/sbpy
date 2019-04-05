# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
================
sbpy Data Module
================

created on June 22, 2017
"""

from copy import deepcopy
from collections import OrderedDict
from numpy import ndarray, array, hstack, ravel
from astropy.table import QTable, Column, vstack
import astropy.units as u

from . import conf

__all__ = ['DataClass', 'mpc_observations', 'sb_search', 'image_search',
           'pds_ferret']


class DataClass():
    """`~sbpy.data.DataClass` serves as the base class for all data
    container classes in `sbpy` in order to provide consistent
    functionality throughout all these classes.

    The core of `~sbpy.data.DataClass` is an `~astropy.table.QTable`
    object (referred to as the `data table` below) - a type of
    `~astropy.table.Table` object that supports the `~astropy.units`
    formalism on a per-column base - which already provides most of
    the required functionality. `~sbpy.data.DataClass` objects can be
    manually generated from dictionaries
    (`~sbpy.data.DataClass.from_dict`), `~numpy.array`-like
    (`~sbpy.data.DataClass.from_array`) objects, or directly from
    another `~astropy.table.QTable` object.

    A few high-level functions for table data access or modification
    are provided; other, more complex modifications can be applied to
    the underlying table object (`~sbpy.data.DataClass.table`) directly.

    """

    def __init__(self, data):
        self._table = QTable()
        # self.altkeys = {}  # dictionary for alternative column names

        if (len(data.items()) == 1 and 'table' in data.keys()):
            # single item provided named 'table' -> already Table object
            self._table = QTable(data['table'])
        else:
            # treat kwargs as dictionary
            for key, val in data.items():
                try:
                    unit = val.unit
                    val = val.value
                except AttributeError:
                    unit = None

                # check if val is already list-like
                try:
                    val[0]
                except (TypeError, IndexError):
                    val = [val]

                self._table[key] = Column(val, unit=unit)

    @classmethod
    def from_dict(cls, data):
        """Create `~sbpy.data.DataClass` object from dictionary or list of
        dictionaries.

        Parameters
        ----------
        data : `~collections.OrderedDict`, dictionary or list (or similar) of
             dictionaries Data that will be ingested in
             `~sbpy.data.DataClass` object.  Each dictionary creates a
             row in the data table. Dictionary keys are used as column
             names; corresponding values must be scalar (cannot be
             lists or arrays). If a list of dictionaries is provided,
             all dictionaries have to provide the same set of keys
             (and units, if used at all).

        Returns
        -------
        `DataClass` object

        Examples
        --------
        >>> import astropy.units as u
        >>> from sbpy.data import Orbit
        >>> orb = Orbit.from_dict({'a': 2.7674*u.au,
        ...                        'e': 0.0756,
        ...                        'i': 10.59321*u.deg})

        Since dictionaries have no specific order, the ordering of the
        column in the example above is not defined. If your data table
        requires a specific order, use an ``OrderedDict``:

        >>> from collections import OrderedDict
        >>> orb = Orbit.from_dict(OrderedDict([('a', 2.7674*u.au),
        ...                                    ('e', 0.0756),
        ...                                    ('i', 10.59321*u.deg)]))
        >>> print(orb)
        <QTable length=1>
           a       e       i
           AU             deg
        float64 float64 float64
        ------- ------- --------
         2.7674  0.0756 10.59321
        >>> print(orb.column_names) # doctest: +SKIP
        <TableColumns names=('a','e','i')>
        >>> print(orb.table['a', 'e', 'i'])
          a      e       i
          AU            deg
        ------ ------ --------
        2.7674 0.0756 10.59321

        """
        if isinstance(data, (dict, OrderedDict)):
            return cls(data)
        elif isinstance(data, (list, ndarray, tuple)):
            # build table from first dict and append remaining rows
            tab = cls(data[0])
            for row in data[1:]:
                tab.add_rows(row)
            return tab
        else:
            raise TypeError('this function requires a dictionary or a '
                            'list of dictionaries')

    @classmethod
    def from_array(cls, data, names):
        """Create `~sbpy.data.DataClass` object from list, `~numpy.ndarray`,
        or tuple.

        Parameters
        ----------
        data : list, `~numpy.ndarray`, or tuple
            Data that will be ingested in `DataClass` object. A one
            dimensional sequence will be interpreted as a single row. Each
            element that is itself a sequence will be interpreted as a
            column.
        names : list
            Column names, must have the same number of names as data columns.

        Returns
        -------
        `DataClass` object

        Examples
        --------
        >>> from sbpy.data import DataClass
        >>> import astropy.units as u
        >>> dat = DataClass.from_array([[1, 2, 3]*u.deg,
        ...                             [4, 5, 6]*u.km,
        ...                             ['a', 'b', 'c']],
        ...                            names=('a', 'b', 'c'))
        >>> print(dat.table)
         a   b   c
        deg  km
        --- --- ---
        1.0 4.0   a
        2.0 5.0   b
        3.0 6.0   c

        """

        if isinstance(data, (list, ndarray, tuple)):
            return cls.from_dict(OrderedDict(zip(names, data)))
        else:
            raise TypeError('this function requires a list, tuple or a '
                            'numpy array')

    @classmethod
    def from_table(cls, data):
        """Create `DataClass` object from `~astropy.table.Table` or
        `astropy.table.QTable` object.

        Parameters
        ----------
        data : astropy `Table` object, mandatory
             Data that will be ingested in `DataClass` object.

        Returns
        -------
        `DataClass` object

        Examples
        --------
        >>> from astropy.table import QTable
        >>> import astropy.units as u
        >>> from sbpy.data import DataClass
        >>> tab = QTable([[1,2,3]*u.kg,
        ...               [4,5,6]*u.m/u.s,],
        ...              names=['mass', 'velocity'])
        >>> dat = DataClass.from_table(tab)
        >>> print(dat.table)
        mass velocity
         kg   m / s
        ---- --------
         1.0      4.0
         2.0      5.0
         3.0      6.0
        """

        return cls({'table': data})

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Create `DataClass` object from a file using
        `~astropy.table.Table.read`.

        Parameters
        ----------
        filename : str
             Name of the file that will be read and parsed.
        **kwargs : additional parameters
             Optional parameters that will be passed on to
             `~astropy.table.Table.read`.

        Returns
        -------
        `DataClass` object

        Notes
        -----
        This function is merely a wrapper around
        `~astropy.table.Table.read`. Please refer to the documentation of
        that function for additional information on optional parameters
        and data formats that are available. Furthermore, note that this
        function is not able to identify units. If you want to work with
        `~astropy.units` you have to assign them manually to the object
        columns.

        Examples
        --------
        >>> from sbpy.data import DataClass

        >>> dat = DataClass.from_file('data.txt',
        ...                           format='ascii') # doctest: +SKIP
        """

        data = QTable.read(filename, **kwargs)

        return cls({'table': data})

    def to_file(self, filename, format='ascii', **kwargs):
        """Write object to a file using
        `~astropy.table.Table.write`.

        Parameters
        ----------
        filename : str
             Name of the file that will be written.
        format : str, optional
             Data format in which the file should be written. Default:
             ``ASCII``
        **kwargs : additional parameters
             Optional parameters that will be passed on to
             `~astropy.table.Table.write`.

        Returns
        -------
        None

        Notes
        -----
        This function is merely a wrapper around
        `~astropy.table.Table.write`. Please refer to the
        documentation of that function for additional information on
        optional parameters and data formats that are
        available. Furthermore, note that this function is not able to
        write unit information to the file.

        Examples
        --------
        >>> from sbpy.data import DataClass
        >>> import astropy.units as u
        >>> dat = DataClass.from_array([[1, 2, 3]*u.deg,
        ...                             [4, 5, 6]*u.km,
        ...                             ['a', 'b', 'c']],
        ...                            names=('a', 'b', 'c'))
        >>> dat.to_file('test.txt')

        """

        self._table.write(filename, format=format, **kwargs)

    def __len__(self):
        """Get number of data elements in _table"""
        return len(self._table)

    def __repr__(self):
        """Return representation of the underlying data table
        (``self._table.__repr__()``)"""
        return self._table.__repr__()

    def __getitem__(self, ident):
        """Return columns or rows from data table (``self._table``); checks
        for and may use alternative field names."""

        # iterable
        if isinstance(ident, (list, tuple, ndarray)):
            if all([isinstance(i, str) for i in ident]):
                # list of column names
                self = self._convert_columns(ident)
                newkeylist = [self._translate_columns(i)[0] for i in ident]
                ident = newkeylist
                # return as new DataClass object
                return self.from_table(self._table[ident])
            # ignore lists of boolean (masks)
            elif all([isinstance(i, bool) for i in ident]):
                pass
            # ignore lists of integers
            elif all([isinstance(i, int) for i in ident]):
                pass
        # individual strings
        elif isinstance(ident, str):
            self = self._convert_columns(ident)
            ident = self._translate_columns(ident)[0]

        # return an element from self_table
        return self._table[ident]

    def _translate_columns(self, target_colnames):
        """Translate target_colnames to the corresponding column names
        present in this object's table. Returns a list of actual column
        names present in this object that corresponds to target_colnames
        (order is preserved). Raises KeyError if not all columns are
        present or one or more columns could not be translated.
        """

        if not isinstance(target_colnames, (list, ndarray, tuple)):
            target_colnames = [target_colnames]

        translated_colnames = deepcopy(target_colnames)
        for idx, colname in enumerate(target_colnames):
            # colname is already a column name in self.table
            if colname in self.column_names:
                continue
            # colname is an alternative column name
            elif colname in sum(conf.fieldnames, []):
                for alt in conf.fieldnames[conf.fieldname_idx[colname]]:
                    # translation available for colname
                    if alt in self.column_names:
                        translated_colnames[idx] = alt
                        break
            # colname is unknown, raise a KeyError
            else:
                raise KeyError('field {:s} not available.'.format(
                    colname))

        return translated_colnames

    def _convert_columns(self, target_colnames):
        """Convert target_colnames, if necessary. Converted columns will be
        added as columns to ``self`` using the field names provided in
        target_colnames. No error is returned by this function if a
        field could not be converted.
        """

        if not isinstance(target_colnames, (list, ndarray, tuple)):
            target_colnames = [target_colnames]

        for colname in target_colnames:
            # ignore, if colname is unknown (KeyError)
            try:
                # ignore if colname has already been converted
                if any([alt in self.column_names for alt
                        in conf.fieldnames[conf.fieldname_idx[colname]]]):
                    continue
                # consider alternative names for colname -> alt
                for alt in conf.fieldnames[conf.fieldname_idx[colname]]:
                    if alt in list(conf.field_eq.keys()):
                        # conversion identified
                        convname = self._translate_columns(
                            list(conf.field_eq[alt].keys())[0])[0]
                        convfunc = list(conf.field_eq[alt].values())[0]
                        if convname in self.column_names:
                            # create new column for the converted field
                            self.add_column(convfunc(self.table[convname]),
                                            colname)
                            break
            except KeyError:
                continue

        return self

    @property
    def table(self):
        """Return `~astropy.table.QTable` object containing all data."""
        return self._table

    @property
    def column_names(self):
        """Return a list of all column names in the data table."""
        return self._table.columns

    def add_rows(self, rows, join_type='inner'):
        """Append additional rows to the existing data table. An individual
        row can be provided in list, tuple, `~numpy.ndarray`, or
        dictionary form. Multiple rows can be provided in the form of
        a list, tuple, or `~numpy.ndarray` of individual
        rows. Multiple rows can also be provided in the form of a
        `~astropy.table.QTable` or another `~sbpy.data.DataClass`
        object. Parameter ``join_type`` defines which columns appear
        in the final output table: ``inner`` only keeps those columns
        that appear in both the original table and the rows to be
        added; ``outer`` will keep all columns and populate some with
        placeholders, if necessary. In case of a list, the list
        elements must be in the same order as the table columns. In
        either case, matching `~astropy.units` must be provided in
        ``rows`` if used in the data table.

        Parameters
        ----------
        rows : list, tuple, `~numpy.ndarray`, dict, or `~collections.OrderedDict`
            Data to be appended to the table; required to have the same
            length as the existing table, as well as the same units.
        join_type : str, optional
            Defines which columns are kept in the output table: ``inner``
            only keeps those columns that appear in both the original
            table and the rows to be added; ``outer`` will keep all
            columns and populate them with placeholders, if necessary.
            Default: ``inner``

        Returns
        -------
        n : int, the total number of rows in the data table

        Examples
        --------
        >>> from sbpy.data import DataClass
        >>> import astropy.units as u
        >>> dat = DataClass.from_array([[1, 2, 3]*u.Unit('m'),
        ...                             [4, 5, 6]*u.m/u.s,
        ...                             ['a', 'b', 'c']],
        ...                            names=('a', 'b', 'c'))
        >>> dat.add_rows({'a': 5*u.m, 'b': 8*u.m/u.s, 'c': 'e'})
        4
        >>> print(dat.table)
         a    b    c
         m  m / s
        --- ----- ---
        1.0   4.0   a
        2.0   5.0   b
        3.0   6.0   c
        5.0   8.0   e
        >>> dat.add_rows(([6*u.m, 9*u.m/u.s, 'f'],
        ...               [7*u.m, 10*u.m/u.s, 'g']))
        6
        >>> dat.add_rows(dat)
        12

        """
        if isinstance(rows, QTable):
            self._table = vstack([self._table, rows], join_type=join_type)
        if isinstance(rows, DataClass):
            self._table = vstack([self._table, rows.table],
                                 join_type=join_type)
        if isinstance(rows, (dict, OrderedDict)):
            try:
                newrow = [rows[colname] for colname in self._table.columns]
            except KeyError as e:
                raise ValueError('data for column {0} missing in row {1}'.
                                 format(e, rows))
            self.add_rows(newrow)
        if isinstance(rows, (list, ndarray, tuple)):
            if (not isinstance(rows[0], (u.quantity.Quantity, float)) and
                    isinstance(rows[0], (dict, OrderedDict,
                                         list, ndarray, tuple))):
                for subrow in rows:
                    self.add_rows(subrow)
            else:
                self._table.add_row(rows)
        return len(self._table)

    def add_column(self, data, name, **kwargs):
        """Append a single column to the current data table. The lenght of
        the input list, `~numpy.ndarray`, or tuple must match the current
        number of rows in the data table.

        Parameters
        ----------
        data : list, `~numpy.ndarray`, or tuple
            Data to be filled into the table; required to have the same
            length as the existing table's number rows.
        name : str
            Name of the new column; must be different from already existing
            column names.
        **kwargs : additional parameters
            Additional optional parameters will be passed on to
            `~astropy.table.Table.add_column`.

        Returns
        -------
        n : int, the total number of columns in the data table

        Examples
        --------
        >>> from sbpy.data import DataClass
        >>> import astropy.units as u
        >>> dat = DataClass.from_array([[1, 2, 3]*u.Unit('m'),
        ...                             [4, 5, 6]*u.m/u.s,
        ...                             ['a', 'b', 'c']],
        ...                            names=('a', 'b', 'c'))
        >>> dat.add_column([10, 20, 30]*u.kg, name='d')
        4
        >>> print(dat.table)
         a    b    c   d
         m  m / s      kg
        --- ----- --- ----
        1.0   4.0   a 10.0
        2.0   5.0   b 20.0
        3.0   6.0   c 30.0
        """

        self._table.add_column(Column(data, name=name), **kwargs)
        return len(self.column_names)

    def expand(self, data, name, unit=None):
        """Expand and reshape `~DataClass` object to include one additional
        column.

        Parameters
        ----------
        data : list or `astropy.units.Quantity` object with iterable
            Data to be added in a new column in form of a one-dimensional
            sequence or a two-dimensional nested sequence. Each element in
            `data`
            corresponds to a row in the existing data table. If an element
            of `data` is a list, the corresponding data table row is
            duplicated by the number of elements in this list. If `data` is
            provided as a flat list and has the same length as the current
            data table, `data` will be simply added as a column to the data
            table and the length of the data table will not change. If
            `data` is provided as a `~astropy.units.Quantity` object, its
            unit is adopted, unless `unit` is specified (not `None`).
        name : str
            Name of the new data column.
        unit : `~astropy.units` object or str, optional
            Unit to be applied to the new column. Default:
            `None`

        Returns
        -------
        None

        Note
        ----
        As a result of this method, the length of the underlying data table
        will be the same as the length of the flattened `data` parameter.

        Examples
        --------
        >>> from sbpy.data import DataClass
        >>> import astropy.units as u
        >>> dat = DataClass.from_array([[1, 2, 3]*u.Unit('m'),
        ...                             [4, 5, 6]*u.m/u.s,
        ...                             ['a', 'b', 'c']],
        ...                            names=('a', 'b', 'c'))
        >>> dat.expand([[1], [2, 3], [4, 5, 6]], name='d', unit='kg')
        >>> print(dat)
        <QTable length=6>
           a       b     c      d
           m     m / s          kg
        float64 float64 str1 float64
        ------- ------- ---- -------
            1.0     4.0    a     1.0
            2.0     5.0    b     2.0
            2.0     5.0    b     3.0
            3.0     6.0    c     4.0
            3.0     6.0    c     5.0
            3.0     6.0    c     6.0

        `dat` was expanded and those rows with multiple values
        in the new column `d` were duplicated to match the order in `d`.

        >>> dat.expand([10, 20, 30, 40, 50, 60], name='e')
        >>> print(dat)
        <QTable length=6>
           a       b     c      d       e
           m     m / s          kg
        float64 float64 str1 float64 float64
        ------- ------- ---- ------- -------
            1.0     4.0    a     1.0    10.0
            2.0     5.0    b     2.0    20.0
            2.0     5.0    b     3.0    30.0
            3.0     6.0    c     4.0    40.0
            3.0     6.0    c     5.0    50.0
            3.0     6.0    c     6.0    60.0

        In this case, the new column data provided to `expand` is flat and
        has the same length as the underlying data table. Hence, the new
        column data is simply added a new column.
        """
        _newtable = None

        # strip units off Quantity objects
        if isinstance(data, u.Quantity):
            unit = data.unit
            data = data.value

        if len(data) != len(self.table):
            raise RuntimeError(('DataClass.expand data parameter must have '
                                'same length as self.table'))

        _newcolumn = array([])
        for i, val in enumerate(data):
            if not isinstance(val, (list, tuple, ndarray)):
                val = [val]
            _newcolumn = hstack([_newcolumn, val])
            # add corresponding row from _table for each element in val
            for j in range(len(val)):
                # initialize new QTable object
                if _newtable is None:
                    _newtable = QTable(self.table[0])
                    continue
                _newtable.add_row(self.table[i])

        # add new column
        _newtable.add_column(Column(_newcolumn, name=name, unit=unit))

        self._table = _newtable


def mpc_observations(targetid):
    """Obtain all available observations of a small body from the Minor
    Planet Center (http://www.minorplanetcenter.net) and provides them in
    the form of an Astropy table.

    Parameters
    ----------
    targetid : str, mandatory
        target identifier

    Returns
    -------
    `~sbpy.data.DataClass` object

    Examples
    --------
    >>> from sbpy.data import mpc_observations
    >>> obs = mpc_observations('ceres')  # doctest: +SKIP

    not yet implemented

    """


def sb_search(field):
    """Use the Skybot service (http://vo.imcce.fr/webservices/skybot/) at
    IMCCE to Identify moving objects potentially present in a registered
    FITS images.

    Parameters
    ----------
    field : string, astropy.io.fits Header object, Primary HDU, or Image HDU
      A FITS image file name, HDU data structure, or header with
      defined WCS

    Returns
    -------
    `~sbpy.data.DataClass` object

    Examples
    --------
    >>> from sbpy.data import sb_search
    >>> objects = sb_search('ceres')  # doctest: +SKIP

    not yet implemented

    """


def image_search(targetid):
    """Use the Solar System Object Image Search function of the Canadian
    Astronomy Data Centre
    (http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/ssois/) to identify
    images with a specific small body in them.

    Parameters
    ----------
    targetid : str, mandatory
        target identifier

    Returns
    -------
    `~sbpy.data.DataClass` object

    Examples
    --------
    >>> from sbpy.data import Misc  # doctest: +SKIP
    >>> images = Misc.image_search('ceres')  # doctest: +SKIP

    not yet implemented

    """


def pds_ferret(targetid):
    """Use the Small Bodies Data Ferret (http://sbntools.psi.edu/ferret/)
    at the Planetary Data System's Small Bodies Node to query for
    information on a specific small body in the PDS.

    Parameters
    ----------
    targetid : str, mandatory
        target identifier

    Returns
    -------
    data : dict
      A hierarchical data object

    Examples
    --------
    >>> from sbpy.data import pds_ferret
    >>> data = pds_ferret('ceres')  # doctest: +SKIP

    not yet implemented

    """
