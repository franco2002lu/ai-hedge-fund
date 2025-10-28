# List available games

> Retrieve a directory of all available NFL games with their PolyRouter IDs for use with the Games API.

## OpenAPI

````yaml openapi.json get /list-games
paths:
  path: /list-games
  method: get
  servers:
    - url: https://api.polyrouter.io/functions/v1
      description: Production server
  request:
    security:
      - title: ApiKeyAuth
        parameters:
          query: {}
          header:
            X-API-Key:
              type: apiKey
              description: API key for authentication
          cookie: {}
    parameters:
      path: {}
      query:
        league:
          schema:
            - type: enum<string>
              enum:
                - nfl
              required: true
              description: League ID (required)
        status:
          schema:
            - type: enum<string>
              enum:
                - not_started
                - live
                - finished
              required: false
              description: Filter by game status
        start_date:
          schema:
            - type: string
              required: false
              description: Filter games after this date (ISO 8601)
              format: date
        end_date:
          schema:
            - type: string
              required: false
              description: Filter games before this date (ISO 8601)
              format: date
        limit:
          schema:
            - type: integer
              required: false
              description: Number of results per page (1-100)
              maximum: 100
              minimum: 1
              default: 10
      header: {}
      cookie: {}
    body: {}
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              games:
                allOf:
                  - type: array
                    items:
                      $ref: '#/components/schemas/GameListItem'
              pagination:
                allOf:
                  - $ref: '#/components/schemas/Pagination'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              games:
                - polyrouter_id: DETvKC20251013
                  prophetx_event_id: 123
                  title: Detroit Lions at Kansas City Chiefs
                  away_team:
                    abbreviation: <string>
                    name: <string>
                  home_team:
                    abbreviation: <string>
                    name: <string>
                  scheduled_at: '2023-11-07T05:31:56Z'
                  status: not_started
                  tournament:
                    id: 123
                    name: <string>
                  sport: <string>
              pagination:
                total: 123
                limit: 123
                offset: 123
                has_more: true
                next_offset: 123
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Games list retrieved successfully
    '400':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Validation error response
    '401':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
        description: Authentication error response
    '500':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Internal server error response
  deprecated: false
  type: path
components:
  schemas:
    Pagination:
      type: object
      required:
        - total
        - limit
        - offset
        - has_more
        - next_offset
      properties:
        total:
          type: number
          description: Total number of results
        limit:
          type: number
          description: Number of results per page
        offset:
          type: number
          description: Current page offset
        has_more:
          type: boolean
          description: Whether more pages exist
        next_offset:
          type: number
          description: Offset for next page
    Meta:
      type: object
      required:
        - request_time
        - cache_hit
        - data_freshness
      properties:
        request_time:
          type: number
          description: Request processing time (ms)
        cache_hit:
          type: boolean
          description: Whether response was served from cache
        data_freshness:
          type: string
          format: date-time
          description: Data freshness timestamp
    Error:
      type: object
      required:
        - code
        - message
        - timestamp
      properties:
        code:
          type: string
          description: Error code
          example: VALIDATION_ERROR
        message:
          type: string
          description: Error message
        details:
          type: object
          description: Additional error details
        timestamp:
          type: string
          format: date-time
          description: Error timestamp
    GameListItem:
      type: object
      required:
        - polyrouter_id
        - title
        - away_team
        - home_team
        - status
      properties:
        polyrouter_id:
          type: string
          description: PolyRouter game ID
          example: DETvKC20251013
        prophetx_event_id:
          type: integer
          description: ProphetX event ID
        title:
          type: string
          description: Game title with team names
          example: Detroit Lions at Kansas City Chiefs
        away_team:
          type: object
          properties:
            abbreviation:
              type: string
              description: Team abbreviation
            name:
              type: string
              description: Full team name
        home_team:
          type: object
          properties:
            abbreviation:
              type: string
              description: Team abbreviation
            name:
              type: string
              description: Full team name
        scheduled_at:
          type: string
          format: date-time
          description: Game start time (ISO 8601 UTC)
        status:
          type: string
          enum:
            - not_started
            - live
            - finished
          description: Game status
        tournament:
          type: object
          properties:
            id:
              type: integer
              description: Tournament ID
            name:
              type: string
              description: Tournament name
        sport:
          type: string
          description: Sport name

````