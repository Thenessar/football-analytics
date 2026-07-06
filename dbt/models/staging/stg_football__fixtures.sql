with bronze as (
    select
        raw_payload,
        response_hash,
        ingested_at_utc
    from {{ source('bronze', 'football_fixtures_raw') }}
),

parsed as (
    select
        from_json(
            raw_payload,
            'struct<response:array<struct<fixture:struct<id:int,date:string,status:struct<short:string,long:string>,venue:struct<name:string,city:string>>,league:struct<id:int,name:string,season:int,country:string>,teams:struct<home:struct<id:int,name:string>,away:struct<id:int,name:string>>,goals:struct<home:int,away:int>>>>'
        ) as payload,
        response_hash,
        ingested_at_utc
    from bronze
),

flattened as (
    select
        explode_outer(payload.response) as fixture_entry,
        response_hash,
        ingested_at_utc
    from parsed
),

typed as (
    select
        cast(fixture_entry.fixture.id as int) as fixture_id,
        to_timestamp(fixture_entry.fixture.date) as fixture_date_utc,
        cast(fixture_entry.league.id as int) as league_id,
        fixture_entry.league.name as league_name,
        cast(fixture_entry.league.season as int) as league_season,
        cast(fixture_entry.teams.home.id as int) as home_team_id,
        fixture_entry.teams.home.name as home_team_name,
        cast(fixture_entry.teams.away.id as int) as away_team_id,
        fixture_entry.teams.away.name as away_team_name,
        cast(fixture_entry.goals.home as int) as home_goals,
        cast(fixture_entry.goals.away as int) as away_goals,
        fixture_entry.fixture.status.short as status_short,
        fixture_entry.fixture.status.long as status_long,
        fixture_entry.fixture.venue.name as venue,
        fixture_entry.league.country as country,
        response_hash,
        ingested_at_utc,
        current_timestamp() as updated_at_utc
    from flattened
    where fixture_entry.fixture.id is not null
),

allowed_leagues as (
    select cast(league_id as int) as league_id
    from {{ ref('senior_mens_international_leagues') }}
),

filtered as (
    select typed.*
    from typed
    inner join allowed_leagues
        on typed.league_id = allowed_leagues.league_id
),

deduped as (
    select *
    from filtered
    -- updated_at_utc is a run timestamp and ties within a run; the bronze
    -- ingestion timestamp is the real recency signal, so a fixture re-fetched
    -- after completion deterministically resolves to its latest status.
    qualify row_number() over (
        partition by fixture_id
        order by ingested_at_utc desc, updated_at_utc desc
    ) = 1
)

select * from deduped
