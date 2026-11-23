"""Fail-closed timestamp evidence for an AUDIT overlay, not replacement training data.

All bounds are aware UTC timestamps. A date-only observation must be supplied as
an explicitly justified upper bound (e.g. local end of day converted to UTC).
Nothing derives availability from tournament start + an invented duration.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

KINDS = {'verified_match', 'verified_event_end', 'documented_upper_bound'}
FAMILIES = {'result', 'statistics', 'ranking', 'profile', 'schedule'}


def timestamp(value):
    if value in (None, ''):
        return None
    value = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Evidence timestamps require an explicit timezone')
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Evidence:
    observation_id: str
    family: str
    event_key: str
    source_reference_date: str
    completion_lower: datetime | None = None
    completion_upper: datetime | None = None
    public_available_upper: datetime | None = None
    provider_recorded_at: datetime | None = None
    kind: str = ''
    source: str = ''
    rationale: str = ''
    retrieved_at: datetime | None = None

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError('Unknown information family')
        for name in ('completion_lower', 'completion_upper', 'public_available_upper',
                     'provider_recorded_at', 'retrieved_at'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, datetime)
                                      or value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f'{name} requires an aware timestamp')
        if (self.completion_lower is not None and self.completion_upper is not None
                and self.completion_lower > self.completion_upper):
            raise ValueError('Reversed completion bounds')
        # Result/statistic availability follows completion. For a loose event-end
        # upper bound, using that later bound is deliberately conservative.
        if (self.public_available_upper is not None and self.completion_lower is not None
                and self.public_available_upper < self.completion_lower):
            raise ValueError('Availability upper bound precedes completion lower bound')

    @classmethod
    def from_row(cls, row):
        values = {name: row.get(name, '') for name in (
            'observation_id', 'family', 'event_key', 'source_reference_date',
            'kind', 'source', 'rationale')}
        for name in ('completion_lower', 'completion_upper', 'public_available_upper',
                     'provider_recorded_at', 'retrieved_at'):
            values[name] = timestamp(row.get(name))
        return cls(**values)


def decision(evidence, *, query_id, query_event, prediction_at, cutoff):
    """Return classification and whether a contribution is permitted.

    Valid only for an end-of-match result/statistic audit. Profile/ranking/schedule
    evidence does not require completion, but still requires public availability.
    Provider archival timestamps are retained and never substituted for public time.
    """
    if prediction_at.tzinfo is None or cutoff.tzinfo is None or cutoff >= prediction_at:
        raise ValueError('Require aware cutoff strictly before prediction time')
    if str(query_id) == evidence.observation_id:
        return 'same_match_excluded', False
    if not query_event or not evidence.event_key:
        return 'unresolved_event_identity', False
    if query_event == evidence.event_key:
        return 'same_event_excluded', False
    if (evidence.family in {'result', 'statistics'}
            and evidence.completion_lower is not None
            and evidence.completion_lower > prediction_at):
        return 'confirmed_future_result', False
    if evidence.kind not in KINDS or not evidence.source or not evidence.rationale:
        return 'unresolved_provenance', False
    upper = evidence.public_available_upper
    if upper is None:
        return 'unresolved_public_availability', False
    if evidence.family in {'result', 'statistics'}:
        if evidence.completion_upper is None:
            return 'unresolved_completion', False
        upper = max(upper, evidence.completion_upper)
    if upper > cutoff:
        # A late upper bound does not prove that the result was actually late.
        return 'not_established_by_cutoff', False
    return 'verified_bound_within_cutoff', True


def legacy_cutoff(prediction_at, lag_days=10):
    if lag_days <= 0:
        raise ValueError('Lag must be positive')
    return prediction_at - timedelta(days=lag_days)
