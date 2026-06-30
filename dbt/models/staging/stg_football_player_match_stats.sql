{{ config(alias='football_player_match_stats') }}

with bronze as (
    select
        fixture_id,
        raw_payload,
        response_hash
    from {{ source('bronze', 'football_match_raw') }}
),

parsed as (
    select
        cast(fixture_id as int) as fixture_id,
        from_json(
            raw_payload,
            'struct<response:array<struct<team:struct<id:int,name:string>,players:array<struct<player:struct<id:int,name:string>,statistics:array<struct<games:struct<minutes:int,position:string>,shots:struct<total:int,on:int>,goals:struct<total:int>>>>>>>>'
        ) as payload,
        response_hash
    from bronze
),

flattened as (
    select
        fixture_id,
        team_entry,
        player_entry,
        stat_entry,
        response_hash
    from parsed
    lateral view outer explode(payload.response) team as team_entry
    lateral view outer explode(team_entry.players) player as player_entry
    lateral view outer explode(player_entry.statistics) stat as stat_entry
),

typed as (
    select
        fixture_id,
        cast(team_entry.team.id as int) as team_id,
        team_entry.team.name as team_name,
        cast(player_entry.player.id as int) as player_id,
        player_entry.player.name as player_name,
        cast(stat_entry.games.minutes as int) as games_minutes,
        stat_entry.games.position as games_position,
        cast(coalesce(stat_entry.shots.total, 0) as int) as shots_total,
        cast(coalesce(stat_entry.shots.`on`, 0) as int) as shots_on,
        cast(coalesce(stat_entry.goals.total, 0) as int) as goals_total,
        response_hash,
        {{ normalize_name('player_entry.player.name') }} as player_name_normalized,
        {{ normalize_name('team_entry.team.name') }} as team_name_normalized,
        current_timestamp() as updated_at_utc
    from flattened
    where player_entry.player.id is not null
),

deduped as (
    select *
    from typed
    qualify row_number() over (
        partition by fixture_id, team_id, player_id
        order by updated_at_utc desc
    ) = 1
)

select * from deduped
