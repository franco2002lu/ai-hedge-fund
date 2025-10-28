# List available futures

> Retrieve a directory of all available sports futures with their IDs and metadata. Use this endpoint to discover future IDs for the Futures API v1.

## OpenAPI

````yaml openapi.json get /list-futures
paths:
  path: /list-futures
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
              required: true
              description: League ID (required)
        future_type:
          schema:
            - type: enum<string>
              enum:
                - championship
                - division
                - conference
              required: false
              description: Filter by future type
        season:
          schema:
            - type: integer
              required: false
              description: Season year
              default: 2025
      header: {}
      cookie: {}
    body: {}
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              data:
                allOf:
                  - type: object
                    properties:
                      futures:
                        type: array
                        items:
                          $ref: '#/components/schemas/FutureListItem'
                      pagination:
                        $ref: '#/components/schemas/Pagination'
                      meta:
                        type: object
                        properties:
                          league:
                            type: string
                          data_freshness:
                            type: string
                            format: date-time
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              data:
                futures:
                  - id: nfl_superbowl_2025
                    future_name: Super Bowl LX Winner
                    league: nfl
                    season: 2025
                    future_type: championship
                    category: team
                    platforms:
                      - <string>
                    metadata:
                      deadline: '2023-11-07T05:31:56Z'
                      description: <string>
                      eligible_teams:
                        - <string>
                pagination:
                  total: 123
                  limit: 123
                  offset: 123
                  has_more: true
                  next_offset: 123
                meta:
                  league: <string>
                  data_freshness: '2023-11-07T05:31:56Z'
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Futures list retrieved successfully
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
    FutureListItem:
      type: object
      required:
        - id
        - future_name
        - league
        - season
        - future_type
        - category
        - platforms
      properties:
        id:
          type: string
          description: Future identifier
          example: nfl_superbowl_2025
        future_name:
          type: string
          description: Full future name
          example: Super Bowl LX Winner
        league:
          type: string
          description: League identifier
          example: nfl
        season:
          type: integer
          description: Season year
          example: 2025
        future_type:
          type: string
          description: Future type
          enum:
            - championship
            - division
            - conference
          example: championship
        category:
          type: string
          description: Category
          example: team
        platforms:
          type: array
          items:
            type: string
          description: Platforms with configured markets
        metadata:
          type: object
          properties:
            deadline:
              type: string
              format: date-time
              description: Future deadline
            description:
              type: string
              description: Future description
            eligible_teams:
              type: array
              items:
                type: string
              description: Eligible team IDs

````