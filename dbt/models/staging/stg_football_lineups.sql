{{ config(alias='football_lineups') }}

with bronze as (
    select
        fixture_id,
        raw_payload,
        response_hash
    from {{ source('bronze', 'football_lineups_raw') }}
),

parsed as (
    select
        cast(fixture_id as int) as fixture_id,
        from_json(
            raw_payload,
            'struct<response:array<struct<team:struct<id:int,name:string>,formation:string,startXI:array<struct<player:struct<id:int,name:string,number:int,pos:string>>>,substitutes:array<struct<player:struct<id:int,name:string,number:int,pos:string>>>>>>'
        ) as payload,
        response_hash
    from bronze
),

team_rows as (
    select
        fixture_id,
        team_entry,
        response_hash
    from parsed
    lateral view outer explode(payload.response) team as team_entry
),

starters as (
    select
        fixture_id,
        team_entry,
        player_entry,
        true as is_starting,
        response_hash
    from team_rows
    lateral view outer explode(team_entry.startXI) player as player_entry
),

substitutes as (
    select
        fixture_id,
        team_entry,
        player_entry,
        false as is_starting,
        response_hash
    from team_rows
    lateral view outer explode(team_entry.substitutes) player as player_entry
),

flattened as (
    select * from starters
    union all
    select * from substitutes
),

typed as (
    select
        fixture_id,
        cast(team_entry.team.id as int) as team_id,
        team_entry.team.name as team_name,
        cast(player_entry.player.id as int) as player_id,
        player_entry.player.name as player_name,
        player_entry.player.pos as position,
        cast(player_entry.player.number as int) as number,
        is_starting,
        team_entry.formation as formation,
        response_hash,
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
