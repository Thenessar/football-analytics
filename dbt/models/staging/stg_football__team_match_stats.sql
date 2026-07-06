with bronze as (
    select
        fixture_id,
        raw_payload,
        response_hash,
        ingested_at_utc
    from {{ source('bronze', 'football_fixture_statistics_raw') }}
),

parsed as (
    select
        cast(fixture_id as int) as fixture_id,
        from_json(
            raw_payload,
            'struct<response:array<struct<team:struct<id:int,name:string>,statistics:array<struct<type:string,value:string>>>>>'
        ) as payload,
        response_hash,
        ingested_at_utc
    from bronze
),

team_rows as (
    select
        fixture_id,
        team_entry,
        response_hash,
        ingested_at_utc
    from parsed
    lateral view outer explode(payload.response) team as team_entry
),

stat_rows as (
    select
        fixture_id,
        cast(team_entry.team.id as int) as team_id,
        team_entry.team.name as team_name,
        stat_entry.type as stat_type,
        stat_entry.value as stat_value,
        response_hash,
        ingested_at_utc
    from team_rows
    lateral view outer explode(team_entry.statistics) stat as stat_entry
    where team_entry.team.id is not null
),

pivoted as (
    select
        fixture_id,
        team_id,
        team_name,
        response_hash,
        ingested_at_utc,
        max(case when stat_type = 'Ball Possession' then stat_value end) as possession_raw,
        max(case when stat_type = 'Total Shots' then stat_value end) as shots_total_raw,
        max(case when stat_type = 'Shots on Goal' then stat_value end) as shots_on_raw,
        max(case when stat_type = 'Shots off Goal' then stat_value end) as shots_off_raw,
        max(case when stat_type = 'Blocked Shots' then stat_value end) as shots_blocked_raw,
        max(case when stat_type = 'Shots insidebox' then stat_value end) as shots_inside_box_raw,
        max(case when stat_type = 'Shots outsidebox' then stat_value end) as shots_outside_box_raw,
        max(case when stat_type = 'Total passes' then stat_value end) as passes_total_raw,
        max(case when stat_type = 'Passes accurate' then stat_value end) as passes_accurate_raw,
        max(case when stat_type = 'Passes %' then stat_value end) as passes_pct_raw,
        max(case when stat_type = 'Fouls' then stat_value end) as fouls_raw,
        max(case when stat_type = 'Corner Kicks' then stat_value end) as corners_raw,
        max(case when stat_type = 'Offsides' then stat_value end) as offsides_raw,
        max(case when stat_type = 'Yellow Cards' then stat_value end) as yellow_cards_raw,
        max(case when stat_type = 'Red Cards' then stat_value end) as red_cards_raw,
        max(case when stat_type = 'Goalkeeper Saves' then stat_value end) as goalkeeper_saves_raw,
        max(case when stat_type = 'expected_goals' then stat_value end) as expected_goals_raw
    from stat_rows
    group by fixture_id, team_id, team_name, response_hash, ingested_at_utc
),

typed as (
    select
        fixture_id,
        team_id,
        team_name,
        cast(replace(possession_raw, '%', '') as double) as possession_pct,
        cast(shots_total_raw as int) as shots_total_team,
        cast(shots_on_raw as int) as shots_on_team,
        cast(shots_off_raw as int) as shots_off_team,
        cast(shots_blocked_raw as int) as shots_blocked_team,
        cast(shots_inside_box_raw as int) as shots_inside_box_team,
        cast(shots_outside_box_raw as int) as shots_outside_box_team,
        cast(passes_total_raw as int) as passes_total_team,
        cast(passes_accurate_raw as int) as passes_accurate_team,
        cast(replace(passes_pct_raw, '%', '') as double) as passes_pct_team,
        cast(fouls_raw as int) as fouls_team,
        cast(corners_raw as int) as corners_team,
        cast(offsides_raw as int) as offsides_team,
        cast(yellow_cards_raw as int) as yellow_cards_team,
        cast(red_cards_raw as int) as red_cards_team,
        cast(goalkeeper_saves_raw as int) as goalkeeper_saves_team,
        cast(expected_goals_raw as double) as expected_goals_team,
        response_hash,
        ingested_at_utc,
        {{ normalize_name('team_name') }} as team_name_normalized,
        current_timestamp() as updated_at_utc
    from pivoted
),

deduped as (
    select *
    from typed
    where fixture_id in (select fixture_id from {{ ref('stg_football__fixtures') }})
    qualify row_number() over (
        partition by fixture_id, team_id
        order by ingested_at_utc desc, updated_at_utc desc
    ) = 1
)

select * from deduped
