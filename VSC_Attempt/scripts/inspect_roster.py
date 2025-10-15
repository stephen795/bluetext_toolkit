from src.sleeper_api import SleeperClient
import json

def main():
    c = SleeperClient()
    league_id = '1260098143609966592'
    rosters = c.get_rosters(league_id)
    print('count', len(rosters))
    if rosters:
        print(json.dumps(rosters[0], indent=2))

if __name__ == '__main__':
    main()
