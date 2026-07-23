from astropy.table import Column
job = Gaia.launch_job_async("""SELECT *
                                FROM gaiadr3.nss_two_body_orbit AS g
                                WHERE g.parallax > 1
                                """)

nss_table = job.get_results()

job = Gaia.launch_job_async("""SELECT *
                                FROM gaiadr3.nss_acceleration_astro AS g
                                WHERE g.parallax > 1
                                """)

accel_table = job.get_results()

accel_table.remove_column("corr_vec")
nss_table.remove_column("corr_vec")

nss_table['nss_solution_type'] = Column(nss_table['nss_solution_type'].astype('U30'), name='nss_solution_type')
for colname in accel_table.colnames:
    col = accel_table[colname]

    if col.dtype.kind in ('U', 'S', 'O'):
        if hasattr(col, 'masked') and col.masked:
            col = col.filled('')

        maxlen = max(len(str(x)) for x in col)
        accel_table[colname] = Column(col.astype(f'S{maxlen}'), name=colname)
nss_table.write(f'../BigData/NSSTableKpc.fits', format='fits', overwrite=True)
accel_table.write(f'../BigData/AccelTableKpc.fits', format='fits', overwrite=True)