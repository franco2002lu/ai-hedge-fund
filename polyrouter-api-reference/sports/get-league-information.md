# Get league information

> Retrieve league metadata and team information with platform-specific ID mappings. This endpoint provides canonical team identifiers (PolyRouter IDs) used across all sports endpoints. Supports NFL (fully operational), NBA, NHL, and MLB (limited - registry only).

## OpenAPI

````yaml openapi.json get /league-info
paths:
  path: /league-info
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
                - nba
                - nhl
                - mlb
              required: false
              description: Filter by specific league. Returns all leagues if not specified.
      header: {}
      cookie: {}
    body: {}
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              leagues:
                allOf:
                  - type: array
                    items:
                      $ref: '#/components/schemas/League'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              leagues:
                - id: nfl
                  name: National Football League
                  abbreviation: NFL
                  sport: football
                  season:
                    year: 123
                    start_date: '2023-12-25'
                    end_date: '2023-12-25'
                  teams:
                    - polyrouter_id: PIT
                      name: Pittsburgh Steelers
                      abbreviation: PIT
                      city: Pittsburgh
                      state: PA
                      conference: AFC
                      division: North
                      platform_ids:
                        polymarket: <string>
                        kalshi: <string>
                        prophetx: <string>
                        novig: <string>
                        sxbet: <string>
                      metadata:
                        colors:
                          - <string>
                        founded: 123
                        stadium: <string>
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: League information retrieved successfully
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
    '404':
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
        description: Resource not found error response
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
    League:
      type: object
      required:
        - id
        - name
        - abbreviation
        - sport
      properties:
        id:
          type: string
          description: Unique league identifier
          example: nfl
        name:
          type: string
          description: Full league name
          example: National Football League
        abbreviation:
          type: string
          description: League abbreviation
          example: NFL
        sport:
          type: string
          description: Sport type
          example: football
        season:
          type: object
          properties:
            year:
              type: integer
              description: Season year
            start_date:
              type: string
              format: date
              description: Season start date
            end_date:
              type: string
              format: date
              description: Season end date
        teams:
          type: array
          items:
            $ref: '#/components/schemas/Team'
          description: Array of team objects
    Team:
      type: object
      required:
        - polyrouter_id
        - name
        - abbreviation
      properties:
        polyrouter_id:
          type: string
          description: Canonical PolyRouter team ID
          example: PIT
        name:
          type: string
          description: Full team name
          example: Pittsburgh Steelers
        abbreviation:
          type: string
          description: Official team abbreviation
          example: PIT
        city:
          type: string
          description: Team city/region
          example: Pittsburgh
        state:
          type: string
          description: Team state/province
          example: PA
        conference:
          type: string
          description: Conference
          example: AFC
        division:
          type: string
          description: Division
          example: North
        platform_ids:
          type: object
          properties:
            polymarket:
              type: string
              description: Polymarket team identifier
            kalshi:
              type: string
              description: Kalshi team identifier
            prophetx:
              type: string
              description: ProphetX team identifier
            novig:
              type: string
              description: Novig team identifier
            sxbet:
              type: string
              description: SX.bet team identifier
        metadata:
          type: object
          properties:
            colors:
              type: array
              items:
                type: string
              description: Team colors (hex codes)
            founded:
              type: integer
              description: Year team was founded
            stadium:
              type: string
              description: Home stadium name

````