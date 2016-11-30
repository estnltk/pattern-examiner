from sqlalchemy import create_engine, MetaData, select, and_, func
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from patternexaminer.config import config

engine = create_engine('sqlite:///patternexaminer_public.db', convert_unicode=True)

db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))

Base = declarative_base()
Base.query = db_session.query_property()

def init_db():
    # import all modules here that might define models so that
    # they will be registered properly on the metadata.  Otherwise
    # you will have to import them first before calling init_db()
    import patternexaminer.models as models

    print('dropping pattern_examiner tables, if they exist')
    Base.metadata.drop_all(bind=engine)
    print('creating tables')
    Base.metadata.create_all(bind=engine)
    db_session.commit()


def populate_options():
    print('adding experiment options')
    from patternexaminer.models import Algorithm, Processing, InputType

    db_session.add(Algorithm(name='AgglomerativeClustering'))
    db_session.add(Algorithm(name='SpectralClustering'))
    db_session.add(Algorithm(name='DBSCAN'))
    db_session.add(Algorithm(name='HDBSCAN'))
    db_session.add(Algorithm(name='DecisionTree'))

    db_session.add(Processing(name='Window = +-5'))

    db_session.add(InputType(name='Sports Data'))
    db_session.add(InputType(name='Cluster'))

    db_session.commit()


# Populate the public data table.
def populatePublicData():
    import json
    from patternexaminer.models import PublicData, ResultPublic

    # PublicData.__table__.drop(engine, checkfirst=True)
    # PublicData.__table__.create(engine)

    # ResultPublic.__table__.drop(engine, checkfirst=True)
    # ResultPublic.__table__.create(engine)
    
    
    sports_data = json.load(open('delfi-sports-numbers-wsize5.json'))
    for row in sports_data:
        db_session.add(PublicData(
                    left_context=row['left'], 
                    content=row['number'], 
                    right_context=row['right']))
    db_session.commit()

