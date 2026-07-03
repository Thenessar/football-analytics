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
            'struct<response:array<struct<team:struct<id:int,name:string>,players:array<struct<player:struct<id:int,name:string,photo:string>,statistics:array<struct<games:struct<minutes:int,number:int,position:string,rating:string,captain:boolean,substitute:boolean>,offsides:int,shots:struct<total:int,on:int>,goals:struct<total:int,conceded:int,assists:int,saves:int>,passes:struct<total:int,key:int,accuracy:string>,tackles:struct<total:int,blocks:int,interceptions:int>,duels:struct<total:int,won:int>,dribbles:struct<attempts:int,success:int,past:int>,fouls:struct<drawn:int,committed:int>,cards:struct<yellow:int,red:int>,penalty:struct<won:int,commited:int,scored:int,missed:int,saved:int>>>>>>>>'
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
        player_entry.player.photo as player_photo_url,
        cast(stat_entry.games.minutes as int) as games_minutes,
        cast(stat_entry.games.number as int) as games_number,
        stat_entry.games.position as games_position,
        cast(stat_entry.games.rating as double) as games_rating,
        cast(stat_entry.games.captain as boolean) as games_captain,
        cast(stat_entry.games.substitute as boolean) as games_substitute,
        cast(stat_entry.offsides as int) as offsides,
        cast(coalesce(stat_entry.shots.total, 0) as int) as shots_total,
        cast(coalesce(stat_entry.shots.`on`, 0) as int) as shots_on,
        cast(coalesce(stat_entry.goals.total, 0) as int) as goals_total,
        cast(stat_entry.goals.conceded as int) as goals_conceded,
        cast(stat_entry.goals.assists as int) as goals_assists,
        cast(stat_entry.goals.saves as int) as goals_saves,
        cast(stat_entry.passes.total as int) as passes_total,
        cast(stat_entry.passes.key as int) as passes_key,
        cast(stat_entry.passes.accuracy as int) as passes_accuracy,
        cast(stat_entry.tackles.total as int) as tackles_total,
        cast(stat_entry.tackles.blocks as int) as tackles_blocks,
        cast(stat_entry.tackles.interceptions as int) as tackles_interceptions,
        cast(stat_entry.duels.total as int) as duels_total,
        cast(stat_entry.duels.won as int) as duels_won,
        cast(stat_entry.dribbles.attempts as int) as dribbles_attempts,
        cast(stat_entry.dribbles.success as int) as dribbles_success,
        cast(stat_entry.dribbles.past as int) as dribbles_past,
        cast(stat_entry.fouls.drawn as int) as fouls_drawn,
        cast(stat_entry.fouls.committed as int) as fouls_committed,
        cast(stat_entry.cards.yellow as int) as cards_yellow,
        cast(stat_entry.cards.red as int) as cards_red,
        cast(stat_entry.penalty.won as int) as penalty_won,
        cast(stat_entry.penalty.commited as int) as penalty_committed,
        cast(stat_entry.penalty.scored as int) as penalty_scored,
        cast(stat_entry.penalty.missed as int) as penalty_missed,
        cast(stat_entry.penalty.saved as int) as penalty_saved,
        response_hash,
        {{ normalize_name('player_entry.player.name') }} as player_name_normalized,
        {{ normalize_name('team_entry.team.name') }} as team_name_normalized,
        current_timestamp() as updated_at_utc
    from flattened
    where player_entry.player.id is not null
      and cast(stat_entry.games.minutes as int) > 0
),

deduped as (
    select *
    from typed
    where fixture_id in (select fixture_id from {{ ref('stg_football__fixtures') }})
    qualify row_number() over (
        partition by fixture_id, team_id, player_id
        order by updated_at_utc desc
    ) = 1
)

select * from deduped
