"""Evidence-scoped men's singles registry. Unsupported records never get a default.

Verified means the published governing format is supported, not authentication of
every individual scorecard or proof that no emergency event variation occurred.
Inference across uncollected rulebook years remains explicitly labelled inferred.
The actual played score does not select a rule; it can only raise an audit flag.
"""
import unicodedata


def normalized(value):
    return ''.join(c for c in unicodedata.normalize('NFKD',str(value).lower())
                   if not unicodedata.combining(c)).replace('-',' ').strip()


def resolve(row):
    name=normalized(row['tourney_name'])
    year=int(row['tourney_date'][:4])
    level=row['tourney_level']
    round_=row['round']
    qualifying=round_ in ('Q1','Q2','Q3','Q4')
    best=int(float(row['best_of']))
    result=dict(status='unresolved',ordinary=None,deciding=None,best_of=best,
                sources=[],reason='No supported competition/year/draw mapping',
                draw='qualifying' if qualifying else 'main',format_override=False)
    if not 1990<=year<=2026:
        result['reason']='Outside the documented audit date range'
        return result

    def assign(ordinary,deciding,status,sources,reason,expected=None):
        result.update(ordinary=ordinary,deciding=deciding,status=status,
                      sources=sources,reason=reason)
        if expected is not None and best!=expected:
            result.update(best_of=expected,format_override=True,
                          reason=reason+'; recorded best_of conflicts with published format')
        return result

    if 'next gen' in name:
        if year in (2024,2025):
            return assign('nextgen','nextgen','verified',[f'NEXTGEN_{year}'],
                          'Edition-specific short sets and no-ad games',5)
        result['reason']='Next Gen edition lacks collected edition-specific evidence'
        return result
    if 'laver cup' in name:
        if year in (2024,2025):
            return assign('standard','match_tb10','verified',[f'LAVER_{year}'],
                          'Ten-point match tiebreak replaces the deciding set',3)
        result['reason']='Laver Cup edition lacks collected edition-specific evidence'
        return result

    if level=='G':
        slam=('AO' if 'australian open' in name or name=='doha aus open qualies' else
              'RG' if 'roland garros' in name or 'french open' in name else
              'WIM' if 'wimbledon' in name or 'the championships' in name else
              'US' if name in ('us open','u.s. open','united states open') else None)
        if not slam:
            result['reason']='Unrecognized Grand Slam identity'
            return result
        expected=5 if not qualifying or (slam=='WIM' and round_=='Q3') else 3
        if slam=='AO' and year>=2019:
            return assign('standard','tb10','verified',['AO_2019','GS_2022','GS_2025'],
                          'AO ten-point final-set tiebreak; main and qualifying',expected)
        if year>=2022:
            sources=['GS_2022','GS_2025']
            if slam=='WIM' and qualifying and round_=='Q3':
                sources+=['WIM_QUAL_2024' if year==2024 else 'WIM_QUAL_2025']
            return assign('standard','tb10','verified',sources,
                          'Harmonized final-set ten-point tiebreak',expected)
        if qualifying:
            result['reason']='Earlier qualifying policy needs draw/edition-specific evidence'
            return result
        if slam=='WIM' and year>=2019:
            return assign('standard','wimbledon_12','verified',['WIM_2019'],
                          'Seven-point final-set tiebreak at 12-12',5)
        if slam=='US':
            return assign('standard','standard','verified',['US_HISTORY','GS_2022'],
                          'Historical seven-point final-set tiebreak',5)
        return assign('standard','advantage','inferred',
                      ['AO_2019'] if slam=='AO' else ['WIM_2019','GS_2022'],
                      'Retrospective inference before documented replacement; annual regulations absent',5)

    if level=='O':
        if year==2024 and 'paris' in name:
            return assign('standard','standard','verified',['OLYMPIC_2024'],
                          'Paris 2024 singles: three tiebreak sets',3)
        result['reason']='Earlier Olympic edition/round regulations not collected'
        return result
    if level=='D':
        if year>=2019 and best==3:
            return assign('standard','standard','verified',['DAVIS_HISTORY'],
                          'Post-2019 best-of-three tiebreak singles')
        if 2016<=year<=2018 and best==5:
            return assign('standard','standard','verified',['DAVIS_HISTORY'],
                          '2016-2018 fifth-set tiebreak')
        result['reason']='Davis Cup zone/dead-rubber or earlier format unresolved'
        return result

    if row.get('_source_family')=='atp_matches_futures' or level in ('15','25'):
        if year in (2024,2025,2026):
            status='inferred' if year==2025 else 'verified'
            return assign('standard','match_tb10' if qualifying else 'standard',status,
                          [f'ITF_{year}'],
                          ('ITF 2025 mid-year snapshot; early-year amendment continuity inferred'
                           if year==2025 else
                           'Published ITF singles policy; authorized emergency modifications not enumerated'),3)
        return assign('standard','standard','inferred',['ITF_2024'],
                      'Standard historical Futures policy inferred; annual and qualifying rules absent')

    regular_finals=level=='F' and ('tour finals' in name or 'masters cup' in name)
    if level in ('A','M','C') or regular_finals:
        if year in (2024,2025,2026) and best==3:
            return assign('standard','standard','verified',[f'ATP_{year}'],
                          'Published ATP/Challenger singles policy; known exceptions handled first')
        return assign('standard','standard','inferred',['ATP_2024'],
                      'Historical standard-set continuation inferred; original best_of retained')
    return result


def observed_flags(row,assignment):
    """Post-hoc diagnostics only: never fit, select, or calibrate using outcomes."""
    flags=[]
    if assignment['format_override']:flags.append('best_of_conflict')
    score=row.get('score','')
    if any(t in score.upper() for t in ('RET','W/O','DEF','ABD')):
        flags.append('retirement_or_incomplete_outside_point_model')
    if '[' in score and assignment['deciding']!='match_tb10':
        flags.append('observed_match_tiebreak_conflicts_with_registry')
    return flags
